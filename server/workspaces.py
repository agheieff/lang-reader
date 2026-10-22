"""Profile registry and isolated runtime workspaces."""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from server import settings
from server.schemas import LanguageTag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = settings.load().data_dir
DEFAULT_PROFILE_ID = "es-es"
REGISTRY_VERSION = 1
_PROFILE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_LANGUAGE = TypeAdapter(LanguageTag)


@dataclass(frozen=True)
class Workspace:
    profile_id: str
    label: str
    learning_language: str
    translation_language: str
    directory: Path
    database_path: Path
    relative_directory: str
    relative_database: str


class WorkspaceRegistry:
    def __init__(self, data_dir: Path = DATA_DIR, *, read_only: bool = False) -> None:
        self.data_dir = data_dir.resolve()
        self.read_only = read_only
        self.path = self.data_dir / "registry.json"
        self.lock_path = self.data_dir / "registry.lock"

    def list(self) -> list[Workspace]:
        if self.read_only:
            return [self._workspace(value) for value in self._read_document()["profiles"]]
        with self._lock():
            document = self._load_or_bootstrap()
        return [self._workspace(value) for value in document["profiles"]]

    def selected_id(self) -> str:
        if self.read_only:
            return str(self._read_document()["selected_profile_id"])
        with self._lock():
            return str(self._load_or_bootstrap()["selected_profile_id"])

    def resolve(self, profile_id: str | None = None) -> Workspace:
        requested = profile_id or settings.load().profile or self.selected_id()
        for workspace in self.list():
            if workspace.profile_id == requested:
                return workspace
        raise LookupError(f"profile not found: {requested}")

    def create(
        self,
        profile_id: str,
        *,
        learning_language: str,
        translation_language: str,
        label: str | None = None,
    ) -> Workspace:
        with self.provision(
            profile_id,
            learning_language=learning_language,
            translation_language=translation_language,
            label=label,
        ) as workspace:
            return workspace

    @contextmanager
    def provision(
        self,
        profile_id: str,
        *,
        learning_language: str,
        translation_language: str,
        label: str | None = None,
    ) -> Iterator[Workspace]:
        """Publish one workspace while holding the registry lock, rolling back on failure."""
        self._validate_profile_id(profile_id)
        learning = _LANGUAGE.validate_python(learning_language)
        translation = _LANGUAGE.validate_python(translation_language)
        display_label = (label or learning).strip()
        if not display_label:
            raise ValueError("profile label must not be blank")

        with self._lock():
            document = self._load_or_bootstrap()
            if any(value["id"] == profile_id for value in document["profiles"]):
                raise ValueError(f"profile already exists: {profile_id}")
            entry = {
                "id": profile_id,
                "label": display_label,
                "learning_language": learning,
                "translation_language": translation,
                "directory": f"profiles/{profile_id}",
                "database": f"profiles/{profile_id}/lang.db",
            }
            workspace = self._workspace(entry)
            if workspace.directory.exists():
                raise ValueError(f"workspace directory already exists: {profile_id}")
            document["profiles"].append(entry)
            self._write(document)
            try:
                self._ensure_private_directory(workspace.directory.parent)
                self._ensure_private_directory(workspace.directory)
                yield workspace
            except BaseException:
                document["profiles"] = [
                    value for value in document["profiles"] if value["id"] != profile_id
                ]
                self._write(document)
                raise

    def select(self, profile_id: str) -> Workspace:
        with self._lock():
            document = self._load_or_bootstrap()
            entries = {str(value["id"]): value for value in document["profiles"]}
            if profile_id not in entries:
                raise LookupError(f"profile not found: {profile_id}")
            document["selected_profile_id"] = profile_id
            self._write(document)
            entry = entries[profile_id]
        return self._workspace(entry)

    def delete(
        self,
        profile_id: str,
        *,
        discard: Callable[[Workspace], None],
    ) -> Workspace:
        """Delete one unselected, isolated workspace and unregister it."""
        with self._lock():
            document = self._load_or_bootstrap()
            if document["selected_profile_id"] == profile_id:
                raise ValueError("cannot delete the selected profile; select another profile first")
            entry = next(
                (value for value in document["profiles"] if value["id"] == profile_id),
                None,
            )
            if entry is None:
                raise LookupError(f"profile not found: {profile_id}")
            workspace = self._workspace(entry)
            canonical_directory = self.data_dir / workspace.relative_directory
            if (
                canonical_directory.is_symlink()
                or workspace.database_path.parent != workspace.directory
            ):
                raise ValueError("only profiles with an isolated workspace database can be deleted")

            discard(workspace)
            document["profiles"] = [
                value for value in document["profiles"] if value["id"] != profile_id
            ]
            self._write(document)
        return workspace

    def reset(
        self,
        profile_id: str,
        *,
        clear: Callable[[Workspace], None],
    ) -> Workspace:
        """Clear one unselected workspace while preserving its registry entry."""
        with self._lock():
            document = self._load_or_bootstrap()
            if document["selected_profile_id"] == profile_id:
                raise ValueError("cannot reset the selected profile; select another profile first")
            entry = next(
                (value for value in document["profiles"] if value["id"] == profile_id),
                None,
            )
            if entry is None:
                raise LookupError(f"profile not found: {profile_id}")
            canonical_directory = self.data_dir / str(entry["directory"])
            canonical_database = self.data_dir / str(entry["database"])
            if canonical_directory.is_symlink() or canonical_database.is_symlink():
                raise ValueError("profile workspace and database must not be symbolic links")
            workspace = self._workspace(entry)
            clear(workspace)
        return workspace

    @contextmanager
    def _lock(self) -> Iterator[None]:
        if self.read_only:
            raise ValueError("snapshot registry is read-only")
        self._ensure_private_directory(self.data_dir)
        self.lock_path.touch(mode=0o600, exist_ok=True)
        self.lock_path.chmod(0o600)
        with self.lock_path.open("a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _load_or_bootstrap(self) -> dict[str, Any]:
        if not self.path.exists():
            document = self._bootstrap_document()
            self._write(document)
            return document
        self.path.chmod(0o600)
        document = self._read_document()
        self._harden_workspace_directories(document)
        return document

    def _read_document(self) -> dict[str, Any]:
        """Read an atomically published registry without bootstrapping or hardening it."""
        try:
            value: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("invalid workspace registry JSON") from error
        return self._validate_document(value)

    def _bootstrap_document(self) -> dict[str, Any]:
        """A new data directory starts with one inactive Spanish profile to try the reader."""
        directory = f"profiles/{DEFAULT_PROFILE_ID}"
        self._ensure_private_directory(self.data_dir / "profiles")
        self._ensure_private_directory(self.data_dir / directory)
        return {
            "schema_version": REGISTRY_VERSION,
            "selected_profile_id": DEFAULT_PROFILE_ID,
            "profiles": [
                {
                    "id": DEFAULT_PROFILE_ID,
                    "label": "es-ES",
                    "learning_language": "es-ES",
                    "translation_language": "en",
                    "directory": directory,
                    "database": f"{directory}/lang.db",
                }
            ],
        }

    def _validate_document(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("schema_version") != REGISTRY_VERSION:
            raise ValueError("unsupported workspace registry")
        profiles = value.get("profiles")
        selected = value.get("selected_profile_id")
        if not isinstance(profiles, list) or not profiles or not isinstance(selected, str):
            raise ValueError("invalid workspace registry")
        ids: set[str] = set()
        directories: set[Path] = set()
        databases: set[Path] = set()
        for entry in profiles:
            self._validate_entry(entry)
            profile_id = str(entry["id"])
            if profile_id in ids:
                raise ValueError(f"duplicate profile in registry: {profile_id}")
            ids.add(profile_id)
            directory = self._safe_path(str(entry["directory"]))
            database = self._safe_path(str(entry["database"]))
            if directory in directories:
                raise ValueError("profile workspace directory paths must be unique")
            if database in databases:
                raise ValueError("profile database paths must be unique")
            directories.add(directory)
            databases.add(database)
        if selected not in ids:
            raise ValueError("selected profile is missing from registry")
        return value

    def _validate_entry(self, value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError("invalid profile registry entry")
        required = {
            "id",
            "label",
            "learning_language",
            "translation_language",
            "directory",
            "database",
        }
        if set(value) != required or any(not isinstance(value[key], str) for key in required):
            raise ValueError("invalid profile registry entry")
        self._validate_profile_id(value["id"])
        if not value["label"].strip():
            raise ValueError("profile label must not be blank")
        _LANGUAGE.validate_python(value["learning_language"])
        _LANGUAGE.validate_python(value["translation_language"])
        profile_id = value["id"]
        expected_directory = f"profiles/{profile_id}"
        if value["directory"] != expected_directory:
            raise ValueError(f"profile workspace path is not canonical: {profile_id}")
        if value["database"] != f"{expected_directory}/lang.db":
            raise ValueError(
                f"profile database path is not canonical: {profile_id} (an old shared-database "
                "layout? stop the services and run `lang registry migrate-legacy`)"
            )
        self._safe_path(value["directory"])
        self._safe_path(value["database"])

    def _workspace(self, entry: dict[str, Any]) -> Workspace:
        return Workspace(
            profile_id=str(entry["id"]),
            label=str(entry["label"]),
            learning_language=str(entry["learning_language"]),
            translation_language=str(entry["translation_language"]),
            directory=self._safe_path(str(entry["directory"])),
            database_path=self._safe_path(str(entry["database"])),
            relative_directory=str(entry["directory"]),
            relative_database=str(entry["database"]),
        )

    def _safe_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("workspace paths must stay under the data directory")
        resolved = (self.data_dir / path).resolve()
        if not resolved.is_relative_to(self.data_dir):
            raise ValueError("workspace paths must stay under the data directory")
        return resolved

    def _validate_profile_id(self, profile_id: str) -> None:
        if not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError("profile ID must contain lowercase letters, numbers, and hyphens")

    def _ensure_private_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)

    def _harden_workspace_directories(self, document: dict[str, Any]) -> None:
        for entry in document["profiles"]:
            directory = self._safe_path(str(entry["directory"]))
            self._ensure_private_directory(directory.parent)
            self._ensure_private_directory(directory)

    def _write(self, document: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        os.replace(temporary, self.path)


registry = WorkspaceRegistry()

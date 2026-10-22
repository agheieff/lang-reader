"""Snapshot readers must work without mutating retained originals or accepting events."""

from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import main
from server.db import SCHEMA_VERSION, init_db, session_scope
from server.learning import import_lesson
from server.schemas import LessonDocument
from server.snapshot import SnapshotReader
from server.workspaces import Workspace, WorkspaceRegistry
from tests.conftest import LessonFactory


def fingerprint(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        str(p.relative_to(root)): (
            p.stat().st_mode,
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "directory",
        )
        for p in [root, *root.rglob("*")]
    }


@pytest.fixture
def snapshot(tmp_path: Path, lesson_factory: LessonFactory) -> tuple[Path, Workspace]:
    source = WorkspaceRegistry(tmp_path / "source")
    workspace = source.resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        import_lesson(db, LessonDocument.model_validate(lesson_factory()))
    root = tmp_path / "snapshot"
    target = root / workspace.relative_database
    target.parent.mkdir(parents=True)
    (root / "registry.json").write_bytes(source.path.read_bytes())
    with sqlite3.connect(workspace.database_path) as original, sqlite3.connect(target) as copy:
        original.backup(copy)
        copy.execute("PRAGMA journal_mode=DELETE")
    return root, WorkspaceRegistry(root, read_only=True).resolve("es-es")


def test_snapshot_http_reads_rebuild_only_in_memory(
    snapshot: tuple[Path, Workspace], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, workspace = snapshot
    # Force stale caches. Normal reader replay would write these back to its database.
    with sqlite3.connect(workspace.database_path) as db:
        db.execute("DELETE FROM derived_state")
        db.execute("DELETE FROM lexeme_states")
    before = fingerprint(root)
    monkeypatch.setattr(main, "registry", WorkspaceRegistry(root, read_only=True))
    monkeypatch.setattr(main, "snapshot_reader", SnapshotReader())
    with TestClient(main.app) as client:
        for endpoint in (
            "/api/profiles",
            "/p/es-es",
            "/api/profiles/es-es/reader",
            "/api/profiles/es-es/texts",
            "/api/profiles/es-es/words",
            "/api/profiles/es-es/grammar",
            "/api/profiles/es-es/statistics",
            "/api/profiles/es-es/activation",
            "/api/profiles/es-es/reading-preferences",
            "/api/profiles/es-es/tts",
            "/api/profiles/es-es/lessons/1/audio/status",
        ):
            response = client.get(endpoint)
            assert response.status_code == 200, (endpoint, response.text)
        assert 'data-read-only="true"' in client.get("/p/es-es").text
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            assert client.request(method, "/api/profiles/es-es/events", json={}).status_code == 403
        assert client.get("/api/profiles/missing/reader").status_code == 404
        with ThreadPoolExecutor(max_workers=4) as executor:
            statuses = list(
                executor.map(
                    lambda _: client.get("/api/profiles/es-es/reader").status_code, range(8)
                )
            )
        assert statuses == [200] * 8
    assert fingerprint(root) == before


def test_snapshot_registry_never_bootstraps_or_mutates(tmp_path: Path) -> None:
    registry = WorkspaceRegistry(tmp_path / "missing", read_only=True)
    with pytest.raises(FileNotFoundError):
        registry.list()
    with pytest.raises(ValueError, match="read-only"):
        registry.select("es-es")
    assert not registry.data_dir.exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_snapshot_rejects_live_sidecars(snapshot: tuple[Path, Workspace], suffix: str) -> None:
    _, workspace = snapshot
    Path(str(workspace.database_path) + suffix).touch()
    reader = SnapshotReader()
    with pytest.raises(ValueError, match="without sidecars"), reader.session(workspace):
        pass
    reader.close()


@pytest.mark.parametrize("version", [0, SCHEMA_VERSION - 1, SCHEMA_VERSION + 1])
def test_snapshot_rejects_schema_mismatch(snapshot: tuple[Path, Workspace], version: int) -> None:
    root, workspace = snapshot
    with sqlite3.connect(workspace.database_path) as db:
        db.execute(f"PRAGMA user_version={version}")
    before = fingerprint(root)
    reader = SnapshotReader()
    with pytest.raises(ValueError, match="requires matching code"), reader.session(workspace):
        pass
    reader.close()
    assert fingerprint(root) == before


def test_snapshot_rejects_missing_database(snapshot: tuple[Path, Workspace]) -> None:
    _, workspace = snapshot
    workspace.database_path.unlink()
    reader = SnapshotReader()
    with pytest.raises(ValueError, match="missing"), reader.session(workspace):
        pass
    assert not workspace.database_path.exists()

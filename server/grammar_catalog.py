"""Versioned grammar constructions loaded from tracked TOML catalogs."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[import-not-found]

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from server.packs import pack_files


def normalize_language_tag(value: str) -> str:
    return value.strip().replace("_", "-").casefold()


class GrammarConstruction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(pattern=r"^[a-z]{2,3}:[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=100)
    label: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=600)
    category: str = Field(min_length=1, max_length=60)
    difficulty: float = Field(ge=0.0, le=1.0)
    generation_hint: str | None = Field(default=None, max_length=500)

    @field_validator("label", "description", "category", "generation_hint")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class GrammarCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    id: str = Field(pattern=r"^[a-z]{2,3}(?:-[a-z0-9]+)*$", max_length=40)
    tags: tuple[str, ...] = Field(min_length=1)
    description: str = Field(min_length=1, max_length=300)
    constructions: tuple[GrammarConstruction, ...] = Field(min_length=1, max_length=30)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(normalize_language_tag(tag) for tag in value)
        if any(not tag for tag in normalized):
            raise ValueError("catalog tags must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("catalog tags must be unique")
        return normalized

    @field_validator("description")
    @classmethod
    def strip_description(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_constructions(self) -> GrammarCatalog:
        keys = [construction.key for construction in self.constructions]
        if len(keys) != len(set(keys)):
            raise ValueError("grammar construction keys must be unique within a catalog")
        wrong_namespace = [key for key in keys if not key.startswith(f"{self.id.split('-')[0]}:")]
        if wrong_namespace:
            raise ValueError(
                f"grammar construction keys do not match catalog language: {wrong_namespace}"
            )
        return self

    def construction(self, key: str) -> GrammarConstruction | None:
        return next((item for item in self.constructions if item.key == key), None)


def grammar_catalog(language_tag: str) -> GrammarCatalog:
    return resolve_grammar_catalog(language_tag, grammar_catalogs())


def find_grammar_catalog(language_tag: str) -> GrammarCatalog | None:
    """Resolve an optional catalog without making grammar mandatory for a language pack."""

    try:
        return grammar_catalog(language_tag)
    except LookupError:
        return None


def grammar_catalog_digest(language_tag: str) -> str:
    """Return a stable generation-contract fingerprint for one resolved catalog."""

    serialized = json.dumps(
        grammar_catalog(language_tag).model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def resolve_grammar_catalog(
    language_tag: str, catalogs: tuple[GrammarCatalog, ...]
) -> GrammarCatalog:
    normalized = normalize_language_tag(language_tag)
    matches = [
        (len(tag), catalog)
        for catalog in catalogs
        for tag in catalog.tags
        if normalized == tag or normalized.startswith(f"{tag}-")
    ]
    if not matches:
        raise LookupError(f"no grammar catalog supports learning language: {language_tag}")
    longest = max(length for length, _ in matches)
    most_specific = {catalog.id: catalog for length, catalog in matches if length == longest}
    if len(most_specific) != 1:
        raise ValueError(f"ambiguous grammar catalogs for learning language: {language_tag}")
    return next(iter(most_specific.values()))


def grammar_construction(key: str) -> GrammarConstruction:
    matches = _constructions_by_key(_catalog_signature()).get(key, ())
    if not matches:
        raise LookupError(f"unknown grammar construction: {key}")
    if len(matches) != 1:  # Defensive for callers if catalogs change during a process restart.
        raise ValueError(f"ambiguous grammar construction: {key}")
    return matches[0]


@lru_cache(maxsize=4)
def _constructions_by_key(
    signature: tuple[tuple[str, int, int], ...],
) -> dict[str, tuple[GrammarConstruction, ...]]:
    index: dict[str, list[GrammarConstruction]] = {}
    for catalog in _cached_grammar_catalogs(signature):
        for construction in catalog.constructions:
            index.setdefault(construction.key, []).append(construction)
    return {key: tuple(values) for key, values in index.items()}


def grammar_catalogs() -> tuple[GrammarCatalog, ...]:
    return _cached_grammar_catalogs(_catalog_signature())


@lru_cache(maxsize=16)
def _cached_grammar_catalogs(
    _signature: tuple[tuple[str, int, int], ...],
) -> tuple[GrammarCatalog, ...]:
    return load_grammar_catalogs(pack_files("grammar"))


def load_grammar_catalogs(paths: Iterable[Path]) -> tuple[GrammarCatalog, ...]:
    catalogs: list[GrammarCatalog] = []
    seen_ids: set[str] = set()
    seen_tags: set[str] = set()
    seen_constructions: set[str] = set()
    for path in paths:
        with path.open("rb") as stream:
            raw: Any = tomllib.load(stream)
        catalog = GrammarCatalog.model_validate(raw)
        if catalog.id in seen_ids:
            raise ValueError(f"duplicate grammar catalog id: {catalog.id}")
        duplicate_tags = set(catalog.tags) & seen_tags
        if duplicate_tags:
            raise ValueError(f"duplicate grammar catalog tags: {sorted(duplicate_tags)}")
        keys = {construction.key for construction in catalog.constructions}
        duplicate_constructions = keys & seen_constructions
        if duplicate_constructions:
            raise ValueError(
                f"duplicate grammar construction keys: {sorted(duplicate_constructions)}"
            )
        seen_ids.add(catalog.id)
        seen_tags.update(catalog.tags)
        seen_constructions.update(keys)
        catalogs.append(catalog)
    if not catalogs:
        raise ValueError("no grammar catalogs found (is the packs/ directory present?)")
    return tuple(catalogs)


_SIGNATURE_TTL_SECONDS = 2.0
_catalog_signature_cache: tuple[float, tuple[tuple[str, int, int], ...]] | None = None


def _catalog_signature() -> tuple[tuple[str, int, int], ...]:
    """Detect edited catalogs, stat-ing the directory at most every couple of seconds."""

    global _catalog_signature_cache
    now = time.monotonic()
    if (
        _catalog_signature_cache is None
        or now - _catalog_signature_cache[0] > _SIGNATURE_TTL_SECONDS
    ):
        _catalog_signature_cache = (now, _scan_catalog_signature())
    return _catalog_signature_cache[1]


def _scan_catalog_signature() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in pack_files("grammar")
    )

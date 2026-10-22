"""One-time move of the pre-2026-10 shared database into its profile directory.

Early installs kept the first profile's database at ``<data>/lang.db`` and every other
profile under ``profiles/<id>/``. The registry now requires the isolated layout for all
profiles. Run this with the web service and worker stopped; it copies through SQLite's
backup API (safe for a database in WAL mode), verifies the copy, repoints the registry and
moves the old files aside. Nothing is deleted.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

LEGACY_DATABASE = "lang.db"
SIDECARS = ("", "-wal", "-shm", "-journal")


def migrate_legacy_layout(data_dir: Path) -> dict[str, Any]:
    registry_path = data_dir / "registry.json"
    if not registry_path.is_file():
        return {"state": "current", "moved": []}
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    legacy = [entry for entry in document["profiles"] if entry["database"] == LEGACY_DATABASE]
    if not legacy:
        return {"state": "current", "moved": []}
    source = data_dir / LEGACY_DATABASE
    moved = []
    for entry in legacy:
        target = data_dir / entry["directory"] / "lang.db"
        if target.exists():
            raise ValueError(f"refusing to overwrite {target}")
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _copy_database(source, target)
        entry["database"] = f"{entry['directory']}/lang.db"
        moved.append(entry["id"])
    temporary = registry_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, registry_path)
    aside = data_dir / f"legacy-layout-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    aside.mkdir(mode=0o700)
    for suffix in SIDECARS:
        old = data_dir / f"{LEGACY_DATABASE}{suffix}"
        if old.exists():
            old.rename(aside / old.name)
    return {"state": "migrated", "moved": moved, "kept_in": str(aside)}


def _copy_database(source: Path, target: Path) -> None:
    with sqlite3.connect(f"{source.as_uri()}?mode=rw", uri=True) as origin:
        with sqlite3.connect(target) as copy:
            origin.backup(copy)
        before = origin.execute("PRAGMA user_version").fetchone()[0]
    with sqlite3.connect(target) as copy:
        if copy.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError(f"the copy of {source.name} failed its integrity check")
        if copy.execute("PRAGMA user_version").fetchone()[0] != before:
            raise ValueError(f"the copy of {source.name} has a different schema version")
    target.chmod(0o600)

"""Read frozen, consistent backups without writing to their registry or SQLite files.

Derived projections are rebuilt in RAM using the normal reducers. This is deliberately not a
live-database reader: immutable SQLite ignores WAL, so reject every journal/sidecar and require
an operator-supplied, quiescent snapshot. Never point this at a running workspace.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing, contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from server.db import SCHEMA_VERSION
from server.models import Profile
from server.workspaces import Workspace


class SnapshotReader:
    def __init__(self) -> None:
        self._engines: dict[str, Engine] = {}
        # A profile's single in-memory connection cannot host overlapping transactions.
        self._lock = threading.Lock()

    def _engine(self, workspace: Workspace) -> Engine:
        key = str(workspace.database_path)
        if key in self._engines:
            return self._engines[key]
        path = workspace.database_path
        if not path.is_file():
            raise ValueError(f"snapshot database is missing: {workspace.profile_id}")
        for suffix in ("-wal", "-shm", "-journal"):
            if path.with_name(path.name + suffix).exists():
                raise ValueError("snapshot requires consistent SQLite copies without sidecars")
        before = path.stat()
        memory = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            with closing(
                sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
            ) as source:
                version = source.execute("PRAGMA user_version").fetchone()[0]
                if version != SCHEMA_VERSION:
                    raise ValueError(
                        f"snapshot schema {version} requires matching code ({SCHEMA_VERSION})"
                    )
                source.backup(memory)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("snapshot database changed while opening")
            memory.execute("PRAGMA foreign_keys=ON")
            memory.execute("PRAGMA temp_store=MEMORY")
            engine = create_engine("sqlite://", creator=lambda: memory, poolclass=StaticPool)
            with Session(engine) as db:
                profile = db.get(Profile, 1)
                if profile is None or (
                    profile.learning_language != workspace.learning_language
                    or profile.translation_language != workspace.translation_language
                ):
                    raise ValueError(f"snapshot profile mismatch: {workspace.profile_id}")
        except BaseException:
            memory.close()
            raise
        self._engines[key] = engine
        return engine

    @contextmanager
    def session(self, workspace: Workspace) -> Iterator[Session]:
        with self._lock, Session(self._engine(workspace), expire_on_commit=False) as db:
            db.info["workspace"] = workspace
            yield db

    def close(self) -> None:
        with self._lock:
            for engine in self._engines.values():
                engine.dispose()
            self._engines.clear()

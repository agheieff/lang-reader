"""SQLite engines and sessions routed by profile workspace."""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateIndex, CreateTable

from server.models import Base
from server.workspaces import Workspace, registry

# PRAGMA user_version. Version 1 is the unversioned schema that existed before migrations.
BASELINE_SCHEMA_VERSION = 1
SCHEMA_VERSION = 6
# Derived caches are replayed from append-only evidence, so a derived-table change only drops the
# table; it is recreated below and rebuilt from the evidence on the next read.
MIGRATIONS: dict[int, tuple[str | tuple[str, str, str], ...]] = {
    2: ("DROP TABLE IF EXISTS proficiency_state",),
    3: ("DROP TABLE IF EXISTS lexeme_states", "DROP TABLE IF EXISTS character_states"),
    # (table, column, type): added only when missing, so reruns and fresh schemas are safe.
    4: (("lessons", "known_share_at_import", "FLOAT"),),
    5: (),  # reading_preference_revisions and preference_messages: new tables only
    6: (),  # domain_rejections: new table only
}

_factories: dict[Path, sessionmaker[Session]] = {}
_engines: dict[Path, Engine] = {}
_engine_lock = threading.Lock()


def create_db_engine(url: str, *, database_path: Path | None = None) -> Engine:
    if not url.startswith("sqlite"):
        raise ValueError("only sqlite databases are supported")
    db_engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30.0})

    @event.listens_for(db_engine, "connect")
    def configure_sqlite(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()
        if database_path is not None:
            _secure_sqlite_files(database_path)

    if database_path is not None:

        @event.listens_for(db_engine, "commit")
        @event.listens_for(db_engine, "rollback")
        def secure_after_transaction(_connection: object) -> None:
            _secure_sqlite_files(database_path)

    return db_engine


def _session_factory(workspace: Workspace) -> sessionmaker[Session]:
    path = workspace.database_path.resolve()
    with _engine_lock:
        existing = _factories.get(path)
        if existing is not None:
            return existing
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.parent.chmod(0o700)
        _ensure_private_database(path)
        engine = create_db_engine(f"sqlite:///{path}", database_path=path)
        factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
        _engines[path] = engine
        _factories[path] = factory
        return factory


def _ensure_private_database(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        if not path.is_file():
            raise ValueError(f"workspace database is not a regular file: {path}") from error
    else:
        os.close(descriptor)
    path.chmod(0o600)


def _secure_sqlite_files(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            candidate.chmod(0o600)


def discard_workspace_storage(workspace: Workspace) -> None:
    """Dispose and remove a failed, unpublished profile workspace."""
    path = workspace.database_path.resolve()
    if path.parent != workspace.directory.resolve():
        raise ValueError("only isolated profile workspace databases can be discarded")
    with _engine_lock:
        _factories.pop(path, None)
        engine = _engines.pop(path, None)
    if engine is not None:
        engine.dispose()
    if workspace.directory.exists():
        shutil.rmtree(workspace.directory)


def reset_workspace_storage(workspace: Workspace) -> None:
    """Recreate one profile database and remove its non-database artifacts."""
    reset_db(workspace)
    directory = workspace.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    database = workspace.database_path.resolve()
    preserved_names: set[str] = set()
    if database.parent == directory:
        preserved_names = {
            database.name,
            f"{database.name}-wal",
            f"{database.name}-shm",
        }
    for child in directory.iterdir():
        if child.name in preserved_names:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


@contextmanager
def db_transaction(db: Session) -> Iterator[Session]:
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise


@contextmanager
def session_scope(workspace: Workspace | str | None = None) -> Iterator[Session]:
    """Open a transaction in one explicit profile workspace."""
    resolved = workspace if isinstance(workspace, Workspace) else registry.resolve(workspace)
    with _session_factory(resolved)() as db, db_transaction(db):
        db.info["workspace"] = resolved
        yield db


def init_db(workspace: Workspace | str | None = None) -> Workspace:
    resolved = workspace if isinstance(workspace, Workspace) else registry.resolve(workspace)
    engine = _session_factory(resolved).kw["bind"]
    if not isinstance(engine, Engine):
        raise RuntimeError("workspace database has no engine")
    migrate_database(engine)

    from server.profile_state import ensure_profile

    with session_scope(resolved) as db:
        profile = ensure_profile(
            db,
            learning_language=resolved.learning_language,
            translation_language=resolved.translation_language,
            activated=False,
        )
        if (
            profile.learning_language != resolved.learning_language
            or profile.translation_language != resolved.translation_language
        ):
            raise ValueError(f"profile languages do not match registry: {resolved.profile_id}")
        _backfill_character_states(db)
    return resolved


def _backfill_character_states(db: Session) -> None:
    """Populate the new disposable cache once for an existing Chinese workspace."""

    from sqlalchemy import func, select

    from server.character_learning import rebuild_character_states
    from server.han import character_tracking_available
    from server.models import CharacterState, Lesson, Profile

    profile = db.get(Profile, 1)
    if profile is None or not character_tracking_available(profile.learning_language):
        return
    lessons = db.scalar(select(func.count()).select_from(Lesson)) or 0
    states = db.scalar(select(func.count()).select_from(CharacterState)) or 0
    if lessons and not states:
        rebuild_character_states(db)


def migrate_database(engine: Engine) -> int:
    """Bring one database to SCHEMA_VERSION inside a single exclusive transaction.

    The web service and generation worker start concurrently, so version checks, migrations and
    table creation must not interleave between processes.
    """

    raw = engine.raw_connection()
    try:
        connection: Any = raw.driver_connection
        isolation = connection.isolation_level
        connection.isolation_level = None
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                version = _migrate(connection, engine.dialect)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        finally:
            connection.isolation_level = isolation
    finally:
        raw.close()
    return version


def _migrate(connection: Any, dialect: Dialect) -> int:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if (
        version == 0
        and connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'lessons'"
        ).fetchone()
    ):
        version = BASELINE_SCHEMA_VERSION
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"database schema version {version} is newer than this code ({SCHEMA_VERSION})"
        )
    if version:
        for target in range(version + 1, SCHEMA_VERSION + 1):
            for step in MIGRATIONS.get(target, ()):
                if isinstance(step, str):
                    connection.execute(step)
                    continue
                table_name, column, column_type = step
                existing = connection.execute(f'PRAGMA table_info("{table_name}")')
                columns = {row[1] for row in existing}
                if columns and column not in columns:
                    connection.execute(
                        f'ALTER TABLE "{table_name}" ADD COLUMN "{column}" {column_type}'
                    )
    for table in Base.metadata.sorted_tables:
        connection.execute(str(CreateTable(table, if_not_exists=True).compile(dialect=dialect)))
        for index in sorted(table.indexes, key=lambda item: item.name or ""):
            connection.execute(str(CreateIndex(index, if_not_exists=True).compile(dialect=dialect)))
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return SCHEMA_VERSION


def init_all_databases() -> None:
    for workspace in registry.list():
        init_db(workspace)


def reset_db(workspace: Workspace | str | None = None) -> None:
    resolved = workspace if isinstance(workspace, Workspace) else registry.resolve(workspace)
    engine = _session_factory(resolved).kw["bind"]
    if not isinstance(engine, Engine):
        raise RuntimeError("workspace database has no engine")
    Base.metadata.drop_all(engine)
    init_db(resolved)

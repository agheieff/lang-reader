from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from server.legacy_layout import migrate_legacy_layout
from server.workspaces import WorkspaceRegistry


def _old_layout(root: Path, *, wal: bool = False) -> None:
    """A pre-2026-10 data directory: es-es lives in <root>/lang.db."""
    (root / "profiles" / "es-es").mkdir(parents=True)
    document = {
        "schema_version": 1,
        "selected_profile_id": "es-es",
        "profiles": [
            {
                "id": "es-es",
                "label": "es-ES",
                "learning_language": "es-ES",
                "translation_language": "en",
                "directory": "profiles/es-es",
                "database": "lang.db",
            }
        ],
    }
    (root / "registry.json").write_text(json.dumps(document), encoding="utf-8")
    connection = sqlite3.connect(root / "lang.db")
    if wal:
        connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE kept (value TEXT)")
    connection.execute("INSERT INTO kept VALUES ('history')")
    connection.execute("PRAGMA user_version=5")
    connection.commit()
    if not wal:
        connection.close()
    else:
        # The live case: committed pages are still in the -wal file when the move runs.
        connection.execute("INSERT INTO kept VALUES ('in the wal')")
        connection.commit()
        connection.close()


def test_the_old_layout_is_refused_with_the_command_to_run(tmp_path: Path) -> None:
    _old_layout(tmp_path)
    with pytest.raises(ValueError, match="migrate-legacy"):
        WorkspaceRegistry(tmp_path).list()


@pytest.mark.parametrize("wal", [False, True])
def test_migration_copies_verifies_repoints_and_keeps_the_old_files(
    tmp_path: Path, wal: bool
) -> None:
    _old_layout(tmp_path, wal=wal)

    result = migrate_legacy_layout(tmp_path)

    assert result["state"] == "migrated" and result["moved"] == ["es-es"]
    workspace = WorkspaceRegistry(tmp_path).resolve("es-es")
    assert workspace.relative_database == "profiles/es-es/lang.db"
    with sqlite3.connect(workspace.database_path) as connection:
        values = [row[0] for row in connection.execute("SELECT value FROM kept ORDER BY rowid")]
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
    assert values == (["history", "in the wal"] if wal else ["history"])
    assert not (tmp_path / "lang.db").exists()
    kept = Path(result["kept_in"])
    with sqlite3.connect(kept / "lang.db") as connection:
        assert connection.execute("SELECT count(*) FROM kept").fetchone()[0] == len(values)


def test_migration_is_idempotent_and_never_overwrites(tmp_path: Path) -> None:
    _old_layout(tmp_path)
    migrate_legacy_layout(tmp_path)
    assert migrate_legacy_layout(tmp_path) == {"state": "current", "moved": []}

    _old_layout_again = tmp_path / "other"
    _old_layout_again.mkdir()
    _old_layout(_old_layout_again)
    (_old_layout_again / "profiles" / "es-es" / "lang.db").write_bytes(b"precious")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        migrate_legacy_layout(_old_layout_again)
    assert (_old_layout_again / "lang.db").exists()
    assert (_old_layout_again / "profiles" / "es-es" / "lang.db").read_bytes() == b"precious"


def test_a_fresh_data_directory_needs_no_migration(tmp_path: Path) -> None:
    assert migrate_legacy_layout(tmp_path) == {"state": "current", "moved": []}
    assert (
        WorkspaceRegistry(tmp_path).resolve("es-es").relative_database == "profiles/es-es/lang.db"
    )

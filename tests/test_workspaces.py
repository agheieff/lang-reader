from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from server.db import init_db, session_scope
from server.learning import import_lesson
from server.models import GenerationTask, Lesson
from server.profile_activation import (
    profile_is_active,
)
from server.profile_state import ensure_profile, update_profile
from server.workspaces import WorkspaceRegistry


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _profile_create_args(**changes: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "new_profile_id": "fr",
        "label": None,
        "learning_language": "fr",
        "translation_language": "en",
        "select": False,
        "level": None,
        "difficulty": None,
        "interests": None,
        "text_length": None,
        "known_ratio": None,
        "preference_json": None,
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_profiles_have_isolated_databases(
    tmp_path: Path, lesson_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ARC_LANG_PROFILE", raising=False)
    registry = WorkspaceRegistry(tmp_path)
    spanish = registry.resolve()
    chinese = registry.create(
        "zh-hant",
        label="Chinese (Traditional)",
        learning_language="zh-Hant",
        translation_language="en",
    )
    init_db(spanish)
    init_db(chinese)

    with session_scope(spanish) as db:
        spanish_lesson = import_lesson(db, lesson_factory(key="shared-key"))
        assert ensure_profile(db).learning_language == "es-ES"
    with session_scope(chinese) as db:
        chinese_lesson = import_lesson(
            db,
            lesson_factory(key="shared-key", learning_language="zh-Hant"),
        )
        assert ensure_profile(db).learning_language == "zh-Hant"

    assert spanish_lesson.id == chinese_lesson.id == 1
    assert spanish.database_path != chinese.database_path
    with session_scope(spanish) as db:
        assert db.scalar(select(func.count()).select_from(Lesson)) == 1
    with session_scope(chinese) as db:
        assert db.scalar(select(func.count()).select_from(Lesson)) == 1

    registry.select("zh-hant")
    assert registry.resolve().profile_id == "zh-hant"
    monkeypatch.setenv("ARC_LANG_PROFILE", "es-es")
    assert registry.resolve().profile_id == "es-es"


def test_profile_ids_cannot_escape_data_directory(tmp_path: Path) -> None:
    registry = WorkspaceRegistry(tmp_path)
    with pytest.raises(ValueError, match="profile ID"):
        registry.create(
            "../outside",
            learning_language="fr",
            translation_language="en",
        )


def test_registry_rejects_noncanonical_and_aliased_workspace_paths(tmp_path: Path) -> None:
    registry = WorkspaceRegistry(tmp_path)
    registry.create("one", learning_language="fr", translation_language="en")
    registry.create("two", learning_language="de", translation_language="en")
    document = json.loads(registry.path.read_text(encoding="utf-8"))
    second = next(entry for entry in document["profiles"] if entry["id"] == "two")
    second["database"] = "profiles/one/lang.db"
    registry.path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical"):
        registry.list()

    second["database"] = "profiles/two/lang.db"
    registry.path.write_text(json.dumps(document), encoding="utf-8")
    first_database = tmp_path / "profiles" / "one" / "lang.db"
    first_database.touch()
    (tmp_path / "profiles" / "two" / "lang.db").symlink_to(first_database)

    with pytest.raises(ValueError, match="database paths must be unique"):
        registry.list()


def test_existing_registry_and_sqlite_files_are_private(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    workspace_directory = profiles / "es-es"
    workspace_directory.mkdir(parents=True)
    tmp_path.chmod(0o755)
    profiles.chmod(0o755)
    workspace_directory.chmod(0o755)
    document = {
        "schema_version": 1,
        "selected_profile_id": "es-es",
        "profiles": [
            {
                "id": "es-es",
                "label": "Spanish",
                "learning_language": "es-ES",
                "translation_language": "en",
                "directory": "profiles/es-es",
                "database": "profiles/es-es/lang.db",
            }
        ],
    }
    (tmp_path / "registry.json").write_text(json.dumps(document), encoding="utf-8")
    (tmp_path / "registry.json").chmod(0o644)

    registry = WorkspaceRegistry(tmp_path)
    workspace = registry.resolve()
    init_db(workspace)

    assert _mode(tmp_path) == 0o700
    assert _mode(profiles) == 0o700
    assert _mode(workspace_directory) == 0o700
    assert _mode(registry.path) == 0o600
    assert _mode(registry.lock_path) == 0o600
    assert _mode(workspace.database_path) == 0o600
    for suffix in ("-wal", "-shm"):
        companion = Path(f"{workspace.database_path}{suffix}")
        if companion.exists():
            assert _mode(companion) == 0o600


def test_profile_create_validates_before_publish_and_rolls_back_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    monkeypatch.setattr(cli, "registry", workspace_registry)
    with pytest.raises(json.JSONDecodeError):
        cli.profile_create(_profile_create_args(preference_json="{bad"))
    assert [workspace.profile_id for workspace in workspace_registry.list()] == ["es-es"]
    assert not (tmp_path / "profiles" / "fr").exists()

    def fail_update(*_: Any, **__: Any) -> None:
        raise RuntimeError("configuration failed")

    monkeypatch.setattr(cli, "update_profile", fail_update)
    with pytest.raises(RuntimeError, match="configuration failed"):
        cli.profile_create(_profile_create_args(select=True))
    assert [workspace.profile_id for workspace in workspace_registry.list()] == ["es-es"]
    assert workspace_registry.selected_id() == "es-es"
    assert not (tmp_path / "profiles" / "fr").exists()


def test_profile_create_does_not_delete_preexisting_unregistered_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    preexisting = tmp_path / "profiles" / "fr"
    preexisting.mkdir(parents=True)
    marker = preexisting / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(cli, "registry", workspace_registry)

    with pytest.raises(ValueError, match="workspace directory already exists"):
        cli.profile_create(_profile_create_args())
    assert marker.read_text(encoding="utf-8") == "keep"
    assert [workspace.profile_id for workspace in workspace_registry.list()] == ["es-es"]


def test_cli_profile_create_and_select_leave_profile_inactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    monkeypatch.setattr(cli, "registry", workspace_registry)
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", "1")

    cli.profile_create(_profile_create_args(select=True))
    capsys.readouterr()

    workspace = workspace_registry.resolve("fr")
    with session_scope(workspace) as db:
        assert not profile_is_active(ensure_profile(db))
        assert db.scalar(select(func.count()).select_from(GenerationTask)) == 0


def test_cli_profile_delete_removes_only_the_unselected_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    spanish = workspace_registry.resolve()
    german = workspace_registry.create(
        "de-de",
        label="German",
        learning_language="de",
        translation_language="en",
    )
    init_db(spanish)
    init_db(german)
    spanish_marker = spanish.directory / "keep.txt"
    spanish_marker.write_text("keep", encoding="utf-8")
    german_marker = german.directory / "agent" / "responses" / "lesson.json"
    german_marker.parent.mkdir(parents=True)
    german_marker.write_text("delete", encoding="utf-8")
    german_audio = german.directory / "audio" / "lesson.wav"
    german_audio.parent.mkdir()
    german_audio.write_bytes(b"audio")
    monkeypatch.setattr(cli, "registry", workspace_registry)

    args = cli.build_parser().parse_args(["profile", "delete", "de-de"])
    args.handler(args)

    assert json.loads(capsys.readouterr().out) == {
        "deleted": True,
        "id": "de-de",
        "workspace": "profiles/de-de",
    }
    assert [workspace.profile_id for workspace in workspace_registry.list()] == ["es-es"]
    assert workspace_registry.selected_id() == "es-es"
    assert spanish_marker.read_text(encoding="utf-8") == "keep"
    assert spanish.database_path.exists()
    assert not german.directory.exists()
    with pytest.raises(LookupError, match="profile not found"):
        workspace_registry.resolve("de-de")

    recreated = workspace_registry.create(
        "de-de",
        learning_language="de",
        translation_language="en",
    )
    init_db(recreated)
    assert recreated.database_path.exists()
    assert not german_marker.exists()
    assert not german_audio.exists()


def test_cli_profile_delete_refuses_the_selected_profile_without_removing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    german = workspace_registry.create(
        "de-de",
        learning_language="de",
        translation_language="en",
    )
    init_db(german)
    marker = german.directory / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    workspace_registry.select("de-de")
    monkeypatch.setattr(cli, "registry", workspace_registry)

    args = cli.build_parser().parse_args(["profile", "delete", "de-de"])
    with pytest.raises(ValueError, match="selected profile"):
        args.handler(args)

    assert workspace_registry.selected_id() == "de-de"
    assert workspace_registry.resolve("de-de") == german
    assert marker.read_text(encoding="utf-8") == "keep"


def test_profile_delete_keeps_registry_entry_when_storage_discard_fails(
    tmp_path: Path,
) -> None:
    workspace_registry = WorkspaceRegistry(tmp_path)
    german = workspace_registry.create(
        "de-de",
        learning_language="de",
        translation_language="en",
    )

    def fail_discard(_: Any) -> None:
        raise OSError("storage busy")

    with pytest.raises(OSError, match="storage busy"):
        workspace_registry.delete("de-de", discard=fail_discard)

    assert workspace_registry.resolve("de-de") == german
    assert german.directory.exists()


def test_cli_profile_reset_preserves_isolated_database_and_is_idempotent(
    tmp_path: Path,
    lesson_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    spanish = workspace_registry.resolve()
    german = workspace_registry.create(
        "de-de",
        label="German",
        learning_language="de-DE",
        translation_language="en",
    )
    init_db(spanish)
    init_db(german)
    with session_scope(german) as db:
        import_lesson(
            db,
            lesson_factory(
                key="german-reset",
                learning_language="de-DE",
            ),
        )
    marker = german.directory / "agent" / "draft.json"
    marker.parent.mkdir()
    marker.write_text("old", encoding="utf-8")
    monkeypatch.setattr(cli, "registry", workspace_registry)
    args = cli.build_parser().parse_args(["profile", "reset", "de-de"])

    args.handler(args)
    args.handler(args)

    assert german.database_path.exists()
    assert not marker.exists()
    assert {path.name for path in german.directory.iterdir()} <= {
        "lang.db",
        "lang.db-shm",
        "lang.db-wal",
    }
    with session_scope(german) as db:
        assert not profile_is_active(ensure_profile(db))
        assert db.scalar(select(func.count()).select_from(Lesson)) == 0


def test_cli_profile_reset_refuses_selected_profile_and_unknown_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    selected = workspace_registry.resolve()
    init_db(selected)
    marker = selected.directory / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(cli, "registry", workspace_registry)

    selected_args = cli.build_parser().parse_args(["profile", "reset", "es-es"])
    with pytest.raises(ValueError, match="selected profile"):
        selected_args.handler(selected_args)
    unknown_args = cli.build_parser().parse_args(["profile", "reset", "missing"])
    with pytest.raises(LookupError, match="profile not found"):
        unknown_args.handler(unknown_args)

    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(("auto_generate", "task_available"), [("0", False), ("1", True)])
def test_cli_profile_activate_applies_questionnaire_and_reports_generation_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    auto_generate: str,
    task_available: bool,
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    workspace_registry.create(
        "fr",
        learning_language="fr",
        translation_language="en",
    )
    monkeypatch.setattr(cli, "registry", workspace_registry)
    monkeypatch.setenv("ARC_LANG_AUTO_GENERATE", auto_generate)
    args = cli.build_parser().parse_args(
        [
            "--profile",
            "fr",
            "profile",
            "activate",
            "--starting-point",
            "simple_texts",
            "--confidence",
            "high",
            "--interest",
            "science fiction",
            "--interest",
            " history ",
            "--text-length",
            "240",
        ]
    )

    args.handler(args)

    payload = json.loads(capsys.readouterr().out)
    assert payload["activation"]["active"] is True
    assert payload["activation"]["questionnaire_completed"] is True
    assert payload["activation"]["starting_point"] == "simple_texts"
    assert payload["activation"]["confidence"] == "high"
    assert payload["automatic_generation_enabled"] is (auto_generate == "1")
    assert payload["generation_task_available"] is task_available
    assert (payload["generation_task"] is not None) is task_available

    workspace = workspace_registry.resolve("fr")
    with session_scope(workspace) as db:
        profile = ensure_profile(db)
        assert profile_is_active(profile)
        assert profile.difficulty == pytest.approx(0.25)
        assert profile.interests == ["science fiction", "history"]
        assert profile.preferences["text_length"] == 240
        assert db.scalar(select(func.count()).select_from(GenerationTask)) == int(task_available)


def test_cli_status_recursively_serializes_proficiency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    monkeypatch.setattr(cli, "registry", workspace_registry)
    cli.status(argparse.Namespace(profile_id="es-es"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["profile_id"] == "es-es"
    assert payload["activation"]["active"] is False
    assert payload["activation"]["questionnaire_completed"] is False
    assert payload["proficiency"]["source"] == "unknown"
    assert payload["proficiency"]["status"] == "unstarted"

    cli.generation_ensure(argparse.Namespace(profile_id="es-es"))
    assert json.loads(capsys.readouterr().out) == {"state": "inactive"}


def test_cli_text_length_help_is_compact_and_values_are_bounded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from server import cli

    parser = cli.build_parser()
    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["profile", "activate", "--help"])
    help_text = capsys.readouterr().out

    assert help_exit.value.code == 0
    assert "--text-length TEXT_LENGTH" in help_text
    assert "{50,51,52" not in help_text
    assert parser.parse_args(["profile", "activate", "--text-length", "2000"]).text_length == 2_000
    with pytest.raises(SystemExit):
        parser.parse_args(["profile", "activate", "--text-length", "2001"])
    assert "between 50 and 2000" in capsys.readouterr().err


def test_cli_lesson_list_hides_inactive_legacy_language_rows(
    tmp_path: Path,
    lesson_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from server import cli

    workspace_registry = WorkspaceRegistry(tmp_path)
    workspace = workspace_registry.resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        import_lesson(db, lesson_factory(key="spanish"))
        update_profile(db, {"learning_language": "fr"})
        import_lesson(db, lesson_factory(key="french", learning_language="fr"))
    document = json.loads(workspace_registry.path.read_text(encoding="utf-8"))
    document["profiles"][0]["learning_language"] = "fr"
    workspace_registry.path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(cli, "registry", workspace_registry)

    cli.lesson_list(argparse.Namespace(profile_id="es-es"))

    lessons = json.loads(capsys.readouterr().out)
    assert [lesson["key"] for lesson in lessons] == ["french"]


def test_profile_api_routes_same_ids_to_separate_databases(
    tmp_path: Path,
    lesson_factory: Any,
    event_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from server import main

    registry = WorkspaceRegistry(tmp_path)
    spanish = registry.resolve()
    chinese = registry.create(
        "zh-hant",
        learning_language="zh-Hant",
        translation_language="en",
    )
    init_db(spanish)
    init_db(chinese)
    with session_scope(spanish) as db:
        spanish_lesson = import_lesson(
            db,
            lesson_factory(
                key="same-id",
                terms=[
                    ("es:manana:NOUN", "mañana", "mañana", "NOUN", "morning", 5),
                    ("es:manana:ADV", "mañana", "mañana", "ADV", "tomorrow", 5),
                ],
            ),
        )
    with session_scope(chinese) as db:
        chinese_lesson = import_lesson(
            db,
            lesson_factory(key="same-id", learning_language="zh-Hant"),
        )
    assert spanish_lesson.id == chinese_lesson.id

    monkeypatch.setattr(main, "registry", registry)
    client = TestClient(main.app)
    try:
        spanish_reader = client.get("/api/profiles/es-es/reader")
        chinese_reader = client.get("/api/profiles/zh-hant/reader")
        event = event_factory(
            spanish_lesson.id,
            "lesson.started",
            event_id="same-event-id",
        )
        spanish_event = client.post("/api/profiles/es-es/events", json={"events": [event]})
        chinese_event = client.post("/api/profiles/zh-hant/events", json={"events": [event]})
        legacy_only = event_factory(
            spanish_lesson.id,
            "lesson.started",
            event_id="legacy-pinned-event",
        )
        legacy_event = client.post("/api/profiles/es-es/events", json={"events": [legacy_only]})
        spanish_duplicate = client.post(
            "/api/profiles/es-es/events", json={"events": [legacy_only]}
        )
        chinese_accepts = client.post(
            "/api/profiles/zh-hant/events", json={"events": [legacy_only]}
        )
        isolated_reveal = event_factory(
            spanish_lesson.id,
            "term.revealed",
            event_id="spanish-only-reveal",
            payload={"term_key": "es:manana:NOUN"},
        )
        client.post("/api/profiles/es-es/events", json={"events": [isolated_reveal]})
        spanish_words = client.get("/api/profiles/es-es/words")
        chinese_words = client.get("/api/profiles/zh-hant/words")
    finally:
        client.close()

    assert spanish_reader.json()["profile"]["learning_language"] == "es-ES"
    assert chinese_reader.json()["profile"]["learning_language"] == "zh-Hant"
    assert spanish_event.json()["accepted"] == 1
    assert chinese_event.json()["accepted"] == 1
    assert legacy_event.json()["accepted"] == 1
    assert spanish_duplicate.json()["duplicates"] == 1
    assert chinese_accepts.json()["accepted"] == 1
    assert spanish_words.json()["learning_language"] == "es-ES"
    assert chinese_words.json()["learning_language"] == "zh-Hant"
    spanish_word = next(
        word for word in spanish_words.json()["words"] if word["key"] == "es:manana:NOUN"
    )
    chinese_word = next(
        word for word in chinese_words.json()["words"] if word["key"] == "es:manana:NOUN"
    )
    assert spanish_word["raw_reveal_count"] == 1
    assert chinese_word["raw_reveal_count"] == 0
    assert spanish_word["related_senses"] == [
        {
            "key": "es:manana:ADV",
            "pos": "ADV",
            "gloss": "tomorrow",
            "pronunciation": None,
        }
    ]
    assert chinese_word["related_senses"] == []

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from sqlalchemy import func, select

from server import settings
from server.agent_brief import build_agent_brief
from server.audio_archive import compress_profile_audio
from server.character_learning import rebuild_character_states
from server.content_domains import backfill_domains, rejection_report
from server.db import discard_workspace_storage, init_db, reset_workspace_storage, session_scope
from server.generation_queue import (
    auto_generation_enabled,
    ensure_generation_task,
    retry_generation_task,
)
from server.generation_usage import task_usage
from server.grammar import rebuild_grammar_states
from server.grammar_catalog import find_grammar_catalog, load_grammar_catalogs
from server.language_packs import language_pack, load_language_packs
from server.learning import import_lesson, validate_lesson
from server.legacy_layout import migrate_legacy_layout
from server.lesson_content import body_sentences, lesson_document
from server.lexeme_learning import rebuild_lexeme_states
from server.model_callback import probe_provider
from server.models import (
    CharacterState,
    GenerationTask,
    GrammarState,
    Interaction,
    Lesson,
    LexemeState,
)
from server.packs import pack_files, pack_roots
from server.passwords import hash_password
from server.preference_agent import maybe_update_preferences
from server.profile_activation import (
    QUESTIONNAIRE_SIGMA_FACTOR,
    STARTING_POINT_DIFFICULTY,
    ProfileActivationUpdate,
    activate_profile_settings,
    activation_view,
    profile_is_active,
)
from server.profile_state import (
    build_profile_view,
    ensure_profile,
    rebuild_proficiency_state,
    update_profile,
)
from server.providers import load_config
from server.reading_preferences import preferences_view
from server.schemas import LessonDocument, ProfileUpdate, ProfileView
from server.word_lists import word_list, word_list_summaries
from server.workspaces import Workspace, registry

_JSON_VALUE = TypeAdapter(Any)


def _text_length(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("text length must be an integer") from error
    if not 50 <= parsed <= 2_000:
        raise argparse.ArgumentTypeError("text length must be between 50 and 2000")
    return parsed


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print_json(value: Any) -> None:
    value = _JSON_VALUE.dump_python(value, mode="json")
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _profile_view(db: Any, profile: Any) -> ProfileView:
    return build_profile_view(db, profile)


def _workspace(args: argparse.Namespace) -> Workspace:
    workspace = registry.resolve(args.profile_id)
    init_db(workspace)
    return workspace


def _preferences(args: argparse.Namespace, existing: dict[str, Any]) -> dict[str, Any] | None:
    preferences = dict(existing)
    changed = False
    if args.text_length is not None:
        preferences["text_length"] = args.text_length
        changed = True
    if args.known_ratio is not None:
        preferences["target_known_ratio"] = args.known_ratio
        changed = True
    if args.preference_json is not None:
        extra = json.loads(args.preference_json)
        if not isinstance(extra, dict):
            raise ValueError("--preference-json must contain a JSON object")
        preferences.update(extra)
        changed = True
    return preferences if changed else None


def _profile_patch(args: argparse.Namespace, existing: Any) -> ProfileUpdate:
    return ProfileUpdate(
        learning_language=args.learning_language,
        translation_language=args.translation_language,
        level=args.level,
        difficulty=args.difficulty,
        interests=args.interests,
        preferences=_preferences(args, dict(existing.preferences)),
    )


def profile_list(_: argparse.Namespace) -> None:
    selected = registry.selected_id()
    _print_json(
        [
            {
                "id": workspace.profile_id,
                "label": workspace.label,
                "learning_language": workspace.learning_language,
                "translation_language": workspace.translation_language,
                "selected": workspace.profile_id == selected,
                "workspace": workspace.relative_directory,
                "database": workspace.relative_database,
            }
            for workspace in registry.list()
        ]
    )


def profile_create(args: argparse.Namespace) -> None:
    patch = ProfileUpdate(
        learning_language=args.learning_language,
        translation_language=args.translation_language,
        level=args.level,
        difficulty=args.difficulty,
        interests=args.interests,
        preferences=_preferences(args, {}),
    )
    if patch.learning_language is None or patch.translation_language is None:
        raise ValueError("both profile languages are required")
    workspace: Workspace | None = None
    try:
        with registry.provision(
            args.new_profile_id,
            label=args.label,
            learning_language=patch.learning_language,
            translation_language=patch.translation_language,
        ) as workspace:
            init_db(workspace)
            with session_scope(workspace) as db:
                update_profile(db, patch)
    except BaseException:
        if workspace is not None:
            discard_workspace_storage(workspace)
        raise
    if args.select:
        assert workspace is not None
        registry.select(workspace.profile_id)
    assert workspace is not None
    _print_json({"created": True, "id": workspace.profile_id, "selected": args.select})


def profile_select(args: argparse.Namespace) -> None:
    workspace = registry.select(args.selected_profile_id)
    _print_json({"selected": workspace.profile_id})


def profile_delete(args: argparse.Namespace) -> None:
    workspace = registry.delete(
        args.deleted_profile_id,
        discard=discard_workspace_storage,
    )
    _print_json(
        {
            "deleted": True,
            "id": workspace.profile_id,
            "workspace": workspace.relative_directory,
        }
    )


def profile_reset(args: argparse.Namespace) -> None:
    workspace = registry.reset(
        args.reset_profile_id,
        clear=reset_workspace_storage,
    )
    _print_json(
        {
            "id": workspace.profile_id,
            "reset": True,
            "workspace": workspace.relative_directory,
        }
    )


def profile_show(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        _print_json(_profile_view(db, ensure_profile(db)))


def profile_set(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        profile = ensure_profile(db)
        patch = _profile_patch(args, profile)
        if (
            patch.learning_language is not None
            and patch.learning_language != workspace.learning_language
        ):
            raise ValueError("create a new profile to change the learning language")
        if (
            patch.translation_language is not None
            and patch.translation_language != workspace.translation_language
        ):
            raise ValueError("create a new profile to change the translation language")
        _print_json(_profile_view(db, update_profile(db, patch)))


def profile_activate(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        activation = activate_profile_settings(
            db,
            ProfileActivationUpdate(
                starting_point=args.starting_point,
                confidence=args.confidence,
                interests=args.interests,
                text_length=args.text_length,
            ),
        )
        task = ensure_generation_task(db)
        _print_json(
            {
                "activation": activation,
                "automatic_generation_enabled": auto_generation_enabled(),
                "generation_task_available": task is not None,
                "generation_task": _generation_task_view(task, workspace)
                if task is not None
                else None,
            }
        )


def brief(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        rebuild_lexeme_states(db)
        rebuild_character_states(db)
        rebuild_proficiency_state(db)
        _print_json(build_agent_brief(db))


def lesson_schema(_: argparse.Namespace) -> None:
    _print_json(LessonDocument.model_json_schema())


def lesson_validate(args: argparse.Namespace) -> None:
    document = validate_lesson(_read_json(args.path))
    _print_json(
        {
            "valid": True,
            "key": document.key,
            "terms": len(document.term_catalog()),
            "sentences": len(body_sentences(document)),
            "grammar_occurrences": len(document.grammar_occurrences()),
        }
    )


def lesson_import(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        lesson = import_lesson(
            db,
            _read_json(args.path),
            source_path=str(Path(args.path).resolve()),
            replace=args.replace,
        )
        _print_json({"id": lesson.id, "key": lesson.key, "imported": True})


def lesson_list(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        profile = ensure_profile(db)
        lessons = db.scalars(
            select(Lesson)
            .where(
                Lesson.learning_language == profile.learning_language,
                Lesson.translation_language == profile.translation_language,
            )
            .order_by(Lesson.imported_at, Lesson.id)
        ).all()
        _print_json(
            [
                {
                    "id": lesson.id,
                    "key": lesson.key,
                    "title": document.title,
                    "topic": document.topic,
                    "learning_language": document.learning_language,
                    "translation_language": document.translation_language,
                    "imported_at": lesson.imported_at.isoformat(),
                }
                for lesson in lessons
                for document in [lesson_document(lesson)]
            ]
        )


def rebuild(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        proficiency = rebuild_proficiency_state(db)  # the vocabulary prior uses it
        terms = rebuild_lexeme_states(db)
        characters = rebuild_character_states(db)
        grammar = rebuild_grammar_states(db)
        _print_json(
            {
                "rebuilt": len(terms),
                "characters_rebuilt": len(characters),
                "grammar_rebuilt": len(grammar),
                "proficiency_status": proficiency.status,
                "qualified_calibration_attempts": proficiency.qualified_attempts,
            }
        )


def generation_ensure(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        if not profile_is_active(ensure_profile(db)):
            _print_json({"state": "inactive"})
            return
        task = ensure_generation_task(db, require_enabled=False)
        _print_json(
            _generation_task_view(task, workspace) if task is not None else {"state": "satisfied"}
        )


def generation_list(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        statement = select(GenerationTask).order_by(GenerationTask.id.desc())
        if args.state is not None:
            statement = statement.where(GenerationTask.state == args.state)
        _print_json(
            [_generation_task_view(task, workspace) for task in db.scalars(statement).all()]
        )


def generation_retry(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        _print_json(_generation_task_view(retry_generation_task(db, args.task_id), workspace))


def _generation_task_view(task: GenerationTask, workspace: Workspace) -> dict[str, Any]:
    return {
        "id": task.id,
        "state": task.state,
        "attempts": task.attempts,
        "payload": task.payload,
        "log_path": task.log_path,
        "error": task.error,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "finished_at": task.finished_at.isoformat() if task.finished_at else None,
        "usage": task_usage(workspace.directory / "agent", task.id),
    }


def generation_rejections(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        _print_json(rejection_report(db, workspace))


def content_backfill_domains(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    domains = json.loads(Path(args.file).read_text(encoding="utf-8"))
    if not isinstance(domains, dict):
        raise ValueError("the file must hold a JSON object of lesson key to subject domain")
    with session_scope(workspace) as db:
        _print_json(backfill_domains(db, domains))


def preferences_refresh(args: argparse.Namespace) -> None:
    """Run the agent's normal preference update now instead of waiting for three reactions."""

    # Imported here: the worker module pulls in the whole generation pipeline.
    from server.agent_worker import deferred_by_host
    from server.model_callback import load_callback

    workspace = _workspace(args)
    with session_scope(workspace) as db:
        before = preferences_view(db)
    if deferred_by_host():
        raise ValueError("model work is deferred by the host admission command; try again later")
    if not maybe_update_preferences(workspace, load_callback(), force=True):
        raise ValueError(
            "no new revision was stored: the profile is inactive, the agent failed or was "
            "deferred, or the notes changed while it ran"
        )
    with session_scope(workspace) as db:
        _print_json({"before": before, "after": preferences_view(db)})


def provider_list(_args: argparse.Namespace) -> None:
    config = load_config(settings.load().agent.config_path)
    _print_json(
        {
            "providers": {
                name: provider.model_dump(exclude={"name"}, exclude_none=True)
                for name, provider in config.providers.items()
            },
            "stages": {stage: list(names) for stage, names in config.stages.items()},
        }
    )


def provider_test(args: argparse.Namespace) -> None:
    config = load_config(settings.load().agent.config_path)
    if args.provider not in config.providers:
        raise ValueError(f"unknown provider {args.provider!r}; see `lang provider list`")
    _print_json(probe_provider(config.providers[args.provider]))


def pack_check(args: argparse.Namespace) -> None:
    """Load every discovered pack so a mistake is reported now, not mid-generation."""

    rule_files, grammar_files = pack_files("rules"), pack_files("grammar")
    report: dict[str, Any] = {
        "pack_roots": [str(root) for root in pack_roots()],
        "rules": [
            {"file": str(path), "id": pack.id, "tags": list(pack.tags)}
            for path, pack in zip(rule_files, load_language_packs(rule_files), strict=True)
        ],
        "grammar": [
            {
                "file": str(path),
                "id": catalog.id,
                "tags": list(catalog.tags),
                "constructions": len(catalog.constructions),
            }
            for path, catalog in zip(
                grammar_files, load_grammar_catalogs(grammar_files), strict=True
            )
        ],
        "word_lists": word_list_summaries(),
    }
    report["languages"] = {}
    for tag in args.tags:
        catalog = find_grammar_catalog(tag)
        report["languages"][tag] = {
            "rules": language_pack(tag).id,
            "generation_guidance": len(language_pack(tag).generation_guidance),
            "grammar_catalog": catalog.id if catalog else None,
            "word_list_entries": len(word_list(tag)),
        }
    _print_json(report)


def audio_compress(args: argparse.Namespace) -> None:
    archive = Path(args.archive_dir)
    if not archive.is_absolute():
        raise ValueError("--archive-dir must be an absolute path")
    workspaces = [registry.resolve(args.profile_id)] if args.profile_id else registry.list()
    _print_json([compress_profile_audio(item, archive, f".{args.format}") for item in workspaces])


def auth_hash_password(_args: argparse.Namespace) -> None:
    password = getpass.getpass("New password: ")
    if len(password) < 8:
        raise ValueError("use at least 8 characters")
    if getpass.getpass("Again: ") != password:
        raise ValueError("the passwords differ")
    print(hash_password(password))


def registry_migrate_legacy(_args: argparse.Namespace) -> None:
    _print_json(migrate_legacy_layout(registry.data_dir))


def tts_claim(args: argparse.Namespace) -> None:
    from server.db import init_all_databases
    from server.remote_tts import claim_remote_audio

    init_all_databases()
    _print_json(claim_remote_audio(args.lease_minutes))


def tts_complete(args: argparse.Namespace) -> None:
    from server.remote_tts import MAX_AUDIO_BYTES, complete_remote_audio

    audio = sys.stdin.buffer.read(MAX_AUDIO_BYTES + 1)
    _print_json(
        {
            "relative_path": complete_remote_audio(
                _workspace(args), args.task, audio, claim_token=args.claim
            )
        }
    )


def tts_fail(args: argparse.Namespace) -> None:
    from server.remote_tts import fail_remote_audio

    _print_json(
        {
            "state": fail_remote_audio(
                _workspace(args), args.task, args.error, claim_token=args.claim
            )
        }
    )


def tts_release(args: argparse.Namespace) -> None:
    from server.remote_tts import release_remote_audio

    _print_json(
        {
            "state": release_remote_audio(
                _workspace(args), args.task, args.reason, claim_token=args.claim
            )
        }
    )


def tts_retry(args: argparse.Namespace) -> None:
    from server.remote_tts import retry_failed_audio

    _print_json({"state": retry_failed_audio(_workspace(args), args.task)})


def memory_evaluate(args: argparse.Namespace) -> None:
    from server.memory_evaluation import (
        describe_policy,
        evaluate,
        fit,
        load_history,
        session_order,
    )

    workspace = _workspace(args)
    with session_scope(workspace) as db:
        history = load_history(db)
    result: dict[str, object] = {"production": evaluate(history).as_dict()}
    if args.fit:
        order = session_order(history)
        train, test = set(order[: len(order) // 2]), set(order[len(order) // 2 :])
        fitted, _score = fit(history, sessions=train)
        result["held_out"] = {
            "production": evaluate(history, sessions=test).as_dict(),
            "fitted_on_first_half": evaluate(history, fitted, sessions=test).as_dict(),
            "fitted_policy": describe_policy(fitted),
        }
    _print_json(result)


def status(args: argparse.Namespace) -> None:
    workspace = _workspace(args)
    with session_scope(workspace) as db:
        stored_profile = ensure_profile(db)
        profile = build_profile_view(db, stored_profile)
        _print_json(
            {
                "profile_id": workspace.profile_id,
                "activation": activation_view(stored_profile),
                "lessons": db.scalar(select(func.count()).select_from(Lesson)) or 0,
                "interactions": db.scalar(select(func.count()).select_from(Interaction)) or 0,
                "lexemes": db.scalar(select(func.count()).select_from(LexemeState)) or 0,
                "characters": db.scalar(select(func.count()).select_from(CharacterState)) or 0,
                "grammar_constructions": db.scalar(select(func.count()).select_from(GrammarState))
                or 0,
                "generation_tasks": db.scalar(select(func.count()).select_from(GenerationTask))
                or 0,
                "proficiency": {
                    "difficulty": profile.difficulty,
                    "level": profile.level,
                    **profile.proficiency.model_dump(mode="json"),
                },
            }
        )


def _add_profile_settings(parser: argparse.ArgumentParser, *, languages: bool = True) -> None:
    if languages:
        parser.add_argument("--learning-language")
        parser.add_argument("--translation-language")
    parser.add_argument("--level")
    parser.add_argument("--difficulty", type=float)
    parser.add_argument("--interest", dest="interests", action="append")
    parser.add_argument("--text-length", type=_text_length)
    parser.add_argument("--known-ratio", type=float)
    parser.add_argument("--preference-json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lang", description="Operate the local language reader")
    parser.add_argument("--profile", dest="profile_id", help="Profile ID; overrides the default")
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="Create, select, delete, or configure profiles")
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    profile_commands.add_parser("list").set_defaults(handler=profile_list)
    profile_commands.add_parser("show").set_defaults(handler=profile_show)
    create = profile_commands.add_parser("create")
    create.add_argument("new_profile_id")
    create.add_argument("--label")
    create.add_argument("--learning-language", required=True)
    create.add_argument("--translation-language", required=True)
    create.add_argument("--select", action="store_true")
    _add_profile_settings(create, languages=False)
    create.set_defaults(handler=profile_create)
    select_profile = profile_commands.add_parser("select")
    select_profile.add_argument("selected_profile_id")
    select_profile.set_defaults(handler=profile_select)
    delete_profile = profile_commands.add_parser(
        "delete",
        help="Permanently delete one unselected profile and all of its local files",
    )
    delete_profile.add_argument("deleted_profile_id")
    delete_profile.set_defaults(handler=profile_delete)
    reset_profile = profile_commands.add_parser(
        "reset",
        help="Permanently clear one unselected profile while keeping it available",
    )
    reset_profile.add_argument("reset_profile_id")
    reset_profile.set_defaults(handler=profile_reset)
    profile_set_parser = profile_commands.add_parser("set")
    _add_profile_settings(profile_set_parser)
    profile_set_parser.set_defaults(handler=profile_set)
    profile_activate_parser = profile_commands.add_parser(
        "activate",
        help="Activate a profile and optionally seed its first calibration",
    )
    profile_activate_parser.add_argument(
        "--starting-point",
        choices=tuple(STARTING_POINT_DIFFICULTY),
        default="unsure",
    )
    profile_activate_parser.add_argument(
        "--confidence",
        choices=tuple(QUESTIONNAIRE_SIGMA_FACTOR),
        default="medium",
    )
    profile_activate_parser.add_argument(
        "--interest", dest="interests", action="append", default=[]
    )
    profile_activate_parser.add_argument("--text-length", type=_text_length)
    profile_activate_parser.set_defaults(handler=profile_activate)

    audio = commands.add_parser("audio", help="Maintain cached lesson audio")
    audio_commands = audio.add_subparsers(dest="audio_command", required=True)
    compress = audio_commands.add_parser(
        "compress",
        help="re-encode cached WAV files; originals move to --archive-dir, nothing is deleted",
    )
    compress.add_argument("--archive-dir", required=True)
    compress.add_argument("--format", choices=("opus", "mp3"), default="opus")
    compress.set_defaults(handler=audio_compress)

    provider = commands.add_parser("provider", help="Model providers (the lang-reader.toml file)")
    provider_commands = provider.add_subparsers(dest="provider_command", required=True)
    provider_commands.add_parser(
        "list", help="print the parsed provider file (API keys are never shown)"
    ).set_defaults(handler=provider_list)
    test_parser = provider_commands.add_parser(
        "test", help="send one tiny request to an openai or anthropic provider"
    )
    test_parser.add_argument("provider")
    test_parser.set_defaults(handler=provider_test)

    pack_parser = commands.add_parser("pack", help="Language packs (packs/ and ARC_LANG_PACK_DIRS)")
    pack_commands = pack_parser.add_subparsers(dest="pack_command", required=True)
    check_parser = pack_commands.add_parser(
        "check",
        help="load every pack and show what each language tag resolves to (errors exit 1)",
    )
    check_parser.add_argument("tags", nargs="*", help="language tags to resolve, e.g. fr-CA")
    check_parser.set_defaults(handler=pack_check)

    auth_parser = commands.add_parser("auth", help="Web interface access")
    auth_commands = auth_parser.add_subparsers(dest="auth_command", required=True)
    auth_commands.add_parser(
        "hash-password",
        help="print the ARC_LANG_AUTH_PASSWORD_HASH value for ARC_LANG_AUTH=password",
    ).set_defaults(handler=auth_hash_password)

    registry_parser = commands.add_parser("registry", help="Maintain the profile registry")
    registry_commands = registry_parser.add_subparsers(dest="registry_command", required=True)
    registry_commands.add_parser(
        "migrate-legacy", help="move a pre-2026-10 shared lang.db into its profile directory"
    ).set_defaults(handler=registry_migrate_legacy)

    commands.add_parser("brief").set_defaults(handler=brief)
    commands.add_parser("status").set_defaults(handler=status)
    commands.add_parser("rebuild").set_defaults(handler=rebuild)

    memory = commands.add_parser("memory", help="Score the vocabulary memory model on history")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    evaluate_parser = memory_commands.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--fit", action="store_true", help="also grid-fit on the first half, score the second"
    )
    evaluate_parser.set_defaults(handler=memory_evaluate)

    tts = commands.add_parser("tts", help="Remote audio synthesis (claimed over SSH)")
    tts_commands = tts.add_subparsers(dest="tts_command", required=True)
    claim_parser = tts_commands.add_parser("claim")
    claim_parser.add_argument("--lease-minutes", type=float, default=45.0)
    claim_parser.set_defaults(handler=tts_claim)
    complete_parser = tts_commands.add_parser("complete", help="read WAV bytes from stdin")
    complete_parser.add_argument("--task", type=int, required=True)
    complete_parser.add_argument("--claim", required=True)
    complete_parser.set_defaults(handler=tts_complete)
    fail_parser = tts_commands.add_parser("fail")
    fail_parser.add_argument("--task", type=int, required=True)
    fail_parser.add_argument("--claim", required=True)
    fail_parser.add_argument("--error", required=True)
    fail_parser.set_defaults(handler=tts_fail)
    release_parser = tts_commands.add_parser("release", help="requeue without using an attempt")
    release_parser.add_argument("--task", type=int, required=True)
    release_parser.add_argument("--claim", required=True)
    release_parser.add_argument("--reason", required=True)
    release_parser.set_defaults(handler=tts_release)
    retry_parser = tts_commands.add_parser("retry", help="give a failed task a fresh budget")
    retry_parser.add_argument("--task", type=int, required=True)
    retry_parser.set_defaults(handler=tts_retry)

    lesson = commands.add_parser("lesson", help="Validate and import generated lessons")
    lesson_commands = lesson.add_subparsers(dest="lesson_command", required=True)
    lesson_commands.add_parser("schema").set_defaults(handler=lesson_schema)
    validate_parser = lesson_commands.add_parser("validate")
    validate_parser.add_argument("path")
    validate_parser.set_defaults(handler=lesson_validate)
    import_parser = lesson_commands.add_parser("import")
    import_parser.add_argument("path")
    import_parser.add_argument("--replace", action="store_true")
    import_parser.set_defaults(handler=lesson_import)
    lesson_commands.add_parser("list").set_defaults(handler=lesson_list)

    content = commands.add_parser("content", help="Subject domains of generated texts")
    content_commands = content.add_subparsers(dest="content_command", required=True)
    backfill = content_commands.add_parser(
        "backfill-domains", help="Assign domains to older texts from a JSON file (key: domain)"
    )
    backfill.add_argument("file")
    backfill.set_defaults(handler=content_backfill_domains)

    preferences = commands.add_parser("preferences", help="Reading-preference notes")
    preferences_commands = preferences.add_subparsers(dest="preferences_command", required=True)
    preferences_commands.add_parser(
        "refresh", help="Have the agent recompute the notes from stored signals now"
    ).set_defaults(handler=preferences_refresh)

    generation = commands.add_parser("generation", help="Inspect lesson generation work")
    generation_commands = generation.add_subparsers(dest="generation_command", required=True)
    generation_commands.add_parser("ensure").set_defaults(handler=generation_ensure)
    generation_list_parser = generation_commands.add_parser("list")
    generation_list_parser.add_argument(
        "--state", choices=("pending", "running", "completed", "failed")
    )
    generation_list_parser.set_defaults(handler=generation_list)
    generation_commands.add_parser(
        "rejections", help="Drafts refused for repeating a recent subject domain"
    ).set_defaults(handler=generation_rejections)
    generation_retry_parser = generation_commands.add_parser("retry")
    generation_retry_parser.add_argument("task_id", type=int)
    generation_retry_parser.set_defaults(handler=generation_retry)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "known_ratio", None) is not None and not 0.5 <= args.known_ratio <= 1:
        parser.error("--known-ratio must be between 0.5 and 1")
    if getattr(args, "difficulty", None) is not None and not 0 <= args.difficulty <= 1:
        parser.error("--difficulty must be between 0 and 1")
    try:
        args.handler(args)
    except (OSError, ValueError, LookupError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

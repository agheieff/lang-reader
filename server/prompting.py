"""The chat prompt that carries one generation request to a model.

External commands receive the full canonical request (``generation_callback.canonical_request``).
Chat-style model adapters should send this prompt instead: it leaves out what the model cannot
use (ids, fingerprints, internal scheduling numbers) and orders the JSON so the parts that are
identical across calls of a stage come first. Providers that cache prompt prefixes then reuse
the static start and only pay for the per-task tail.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from server.schemas import (
    GenerationLexicalBatchRequest,
    GenerationLexicalConflictRequest,
    GenerationLexicalUnitRequest,
)

if TYPE_CHECKING:
    from server.generation_callback import GenerationStageRequest

# Host bookkeeping the model neither needs nor should echo.
_HOST_ONLY_KEYS = frozenset(
    {
        "attempt",
        "job_id",
        "lexical_coverage",
        "profile_fingerprint",
        "profile_key",
        "task",
        "task_id",
        "workspace_path",
    }
)
# Identical across the calls of one stage (and mostly across lessons): sent first.
_STATIC_KEYS = (
    "stage",
    "schema_version",
    "instructions",
    "language_guidance",
    "learning_language",
    "translation_language",
    "grammar_catalog",
)
# Differ on every retry or repair: sent last.
_VOLATILE_KEYS = ("previous_failures", "repair_lessons", "repair_result")
_TERM_KEYS = ("key", "lemma", "pos", "gloss", "pronunciation", "frequency_rank")
_PLAN_KEYS = ("archetype", "ending_shape")
_RECENT_KEYS = (
    "title",
    "topic",
    "opening_excerpt",
    "ending_excerpt",
    "completed",
    "rating",
    "feedback",
    "grammar_keys",
    "calibration_lesson",
)


def build_prompt(request: GenerationStageRequest) -> str:
    """The complete task prompt for a chat model; the response schema is supplied separately."""

    document = _prompt_document(request.model_dump(mode="json"), request.stage)
    return (
        f"{_action(request)} for a language-learning reader. Do not use tools, edit files, "
        "inspect the repository, or access a database or network. Return only JSON matching "
        f"the supplied output schema. {_context_instruction(request.stage)} in this request:\n"
        f"{json.dumps(document, ensure_ascii=False, separators=(',', ':'))}"
    )


def _action(request: GenerationStageRequest) -> str:
    if isinstance(request, GenerationLexicalUnitRequest):
        return "Tokenize and define one supplied frozen sentence"
    if isinstance(request, GenerationLexicalBatchRequest):
        return "Tokenize and define each supplied frozen sentence independently"
    if isinstance(request, GenerationLexicalConflictRequest):
        return "Conservatively reconcile ambiguous lexical identities"
    return {
        "legacy": "Generate complete lesson drafts",
        "prose": "Write the frozen lesson prose",
        "lexical": "Tokenize and define the supplied frozen lesson prose",
        "translation": "Translate the supplied frozen lesson prose",
        "grammar": "Annotate grammar in the supplied tokenized lesson prose",
        "preferences": "Maintain the learner's reading-preference notes",
    }[request.stage]


def _context_instruction(stage: str) -> str:
    if stage in {"legacy", "prose"}:
        return "Honor the profile, learning evidence, content plan, and instructions"
    return "Honor the frozen dependencies, task data, and instructions"


def _prompt_document(request: dict[str, Any], stage: str) -> dict[str, Any]:
    body = {key: value for key, value in request.items() if key not in _HOST_ONLY_KEYS}
    brief = body.get("brief")
    if isinstance(brief, dict):
        body["brief"] = _slim_brief(brief, stage)
    if isinstance(body.get("known_terms"), list):
        body["known_terms"] = [_pick(term, _TERM_KEYS) for term in body["known_terms"]]
    return _order(body)


def _slim_brief(brief: dict[str, Any], stage: str) -> dict[str, Any]:
    slim = {key: value for key, value in brief.items() if key != "generated_at"}
    term_keys = (*_TERM_KEYS, "reason")
    slim["priority_terms"] = [_pick(term, term_keys) for term in brief["priority_terms"]]
    slim["recent_lessons"] = [_recent_lesson(lesson) for lesson in brief["recent_lessons"]]
    if stage == "prose":
        # Prose chooses from the ranked priority_grammar pool; the full catalog is the
        # annotation stages' business.
        slim.pop("grammar_catalog", None)
    return slim


def _recent_lesson(lesson: dict[str, Any]) -> dict[str, Any]:
    kept = _pick(lesson, _RECENT_KEYS)
    plan = (lesson.get("metadata") or {}).get("content_plan")
    if isinstance(plan, dict):
        kept["content_plan"] = _pick(plan, _PLAN_KEYS)
    return kept


def _pick(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source and not _empty(source[key])}


def _empty(value: Any) -> bool:
    return value is None or (isinstance(value, (list, dict, str)) and not value)


def _order(body: dict[str, Any]) -> dict[str, Any]:
    first = [key for key in _STATIC_KEYS if key in body]
    last = [key for key in _VOLATILE_KEYS if key in body]
    middle = sorted(set(body) - set(first) - set(last))
    ordered = {key: body[key] for key in (*first, *middle, *last)}
    return {key: value for key, value in ordered.items() if not _empty(value)}

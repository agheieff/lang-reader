"""Canonical accessors for immutable lesson content."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator

from server.models import Lesson
from server.schemas import LessonDocument, LessonRun, LessonSentence

# Parsed documents are shared across sessions and requests, keyed by payload content hash; every
# derived replay reads all lessons, so re-validating them per request dominated its cost.
_MAX_CACHED_DOCUMENTS = 4096
_documents: OrderedDict[str, LessonDocument] = OrderedDict()
_fingerprints: dict[int, str] = {}
_documents_lock = threading.Lock()


def lesson_document(lesson: Lesson) -> LessonDocument:
    """Validate the payload that owns all reader-visible lesson content.

    Treat the returned document as immutable: it is shared by every caller with the same payload.
    ``content_hash`` changes in lockstep with ``payload`` on the lesson-replacement path.
    """

    content_hash = lesson.content_hash
    if content_hash is None:
        return LessonDocument.model_validate(lesson.payload)
    with _documents_lock:
        cached = _documents.get(content_hash)
        if cached is not None:
            _documents.move_to_end(content_hash)
            return cached
    document = LessonDocument.model_validate(lesson.payload)
    with _documents_lock:
        existing = _documents.get(content_hash)
        if existing is not None:
            return existing
        _documents[content_hash] = document
        _fingerprints[id(document)] = content_hash
        while len(_documents) > _MAX_CACHED_DOCUMENTS:
            _, evicted = _documents.popitem(last=False)
            _fingerprints.pop(id(evicted), None)
    return document


def document_fingerprint(document: LessonDocument) -> str | None:
    """Return the content hash of a cached document, or ``None`` for an uncached one."""

    with _documents_lock:
        return _fingerprints.get(id(document))


def body_sentences(document: LessonDocument) -> tuple[LessonSentence, ...]:
    return tuple(sentence for block in document.blocks for sentence in block.sentences)


def all_sentences(document: LessonDocument) -> tuple[LessonSentence, ...]:
    title = (document.title_sentence,) if document.title_sentence is not None else ()
    return (*title, *body_sentences(document))


def lexical_runs(document: LessonDocument) -> Iterator[LessonRun]:
    return (
        run for sentence in all_sentences(document) for run in sentence.runs if run.term is not None
    )


def lexical_token_count(document: LessonDocument) -> int:
    return sum(1 for _run in lexical_runs(document))

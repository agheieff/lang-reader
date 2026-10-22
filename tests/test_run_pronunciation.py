from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session

from server.learning import import_lesson
from server.schemas import LessonRun, LessonTerm


def term() -> LessonTerm:
    return LessonTerm(
        key="zh:行:VERB",
        lemma="行",
        pos="verb",
        gloss="to go",
        pronunciation="háng",
    )


def test_run_pronunciation_is_optional_and_contextual() -> None:
    legacy = LessonRun(text="行", term=term())
    contextual = LessonRun(text="行", term=term(), pronunciation="  xíng  ")

    assert legacy.pronunciation is None
    assert contextual.pronunciation == "xíng"
    assert contextual.model_dump()["pronunciation"] == "xíng"


@pytest.mark.parametrize("pronunciation", ["", "  \t  "])
def test_run_pronunciation_must_not_be_blank(pronunciation: str) -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        LessonRun(text="行", term=term(), pronunciation=pronunciation)


def test_reader_api_preserves_contextual_run_pronunciation(
    api_client: TestClient, db: Session, lesson_factory: Any
) -> None:
    payload = lesson_factory()
    payload["blocks"][0]["sentences"][0]["runs"][1]["pronunciation"] = "xíng"
    import_lesson(db, payload)

    response = api_client.get("/api/profiles/es-es/reader")

    assert response.status_code == 200
    run = response.json()["lesson"]["blocks"][0]["sentences"][0]["runs"][1]
    assert run["pronunciation"] == "xíng"

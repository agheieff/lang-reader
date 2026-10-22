from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from server.learning import import_lesson


def test_health_root_and_empty_reader(api_client: TestClient) -> None:
    health = api_client.get("/api/health")
    root = api_client.get("/")
    reader = api_client.get("/api/profiles/es-es/reader")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": "0.2.0"}
    assert root.status_code == 200
    assert "Lang Reader" in root.text
    assert 'id="skip-lesson-button"' in root.text
    assert 'id="skip-lesson-completion-button"' in root.text
    assert "Optional — your answer helps shape the next text." in root.text
    assert reader.status_code == 200
    assert reader.json()["lesson"] is None
    assert reader.json()["profile"]["learning_language"] == "es-ES"
    assert reader.json()["profile"]["level"] == "B1"
    assert reader.json()["profile"]["difficulty"] == 5 / 12
    assert reader.json()["profile"]["proficiency"]["source"] == "unknown"


def test_settings_is_a_profile_scoped_page(api_client: TestClient) -> None:
    response = api_client.get("/p/es-es", params={"view": "settings"})

    assert response.status_code == 200
    assert 'data-initial-view="settings"' in response.text
    assert 'id="settings-view"' in response.text
    assert 'aria-current="page"' in response.text


def test_texts_is_a_profile_scoped_page(api_client: TestClient) -> None:
    response = api_client.get("/p/es-es", params={"view": "texts"})

    assert response.status_code == 200
    assert 'data-initial-view="texts"' in response.text
    assert 'id="texts-view"' in response.text
    assert 'id="texts-preparing-group"' in response.text
    assert 'id="texts-preparing-list"' in response.text
    assert 'aria-live="polite"' in response.text


def test_statistics_is_an_enabled_profile_scoped_page(api_client: TestClient) -> None:
    response = api_client.get("/p/es-es", params={"view": "statistics"})

    assert response.status_code == 200
    assert 'data-initial-view="statistics"' in response.text
    assert 'id="statistics-view"' in response.text
    assert 'href="#statistics-heading"' in response.text
    assert 'href="/p/es-es?view=statistics"' in response.text
    assert "Statistics, coming soon" not in response.text
    assert 'aria-current="page"' in response.text


def test_grammar_page_and_profile_api_show_only_opened_constructions(
    api_client: TestClient,
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    page = api_client.get("/p/es-es", params={"view": "grammar"})
    assert page.status_code == 200
    assert 'data-initial-view="grammar"' in page.text
    assert 'id="grammar-view"' in page.text

    payload = lesson_factory(key="api-grammar", targets=[])
    payload["blocks"][0]["sentences"][0]["grammar"] = [
        {
            "key": "api-grammar:present",
            "construction_key": "es:present-indicative",
            "run_start": 0,
            "run_end": 2,
        }
    ]
    lesson = import_lesson(db, payload)
    unopened = api_client.get("/api/profiles/es-es/grammar")
    assert unopened.status_code == 200
    assert unopened.json()["constructions"] == []

    opened = api_client.post(
        "/api/profiles/es-es/events",
        json={"events": [event_factory(lesson.id, "lesson.started", event_id="open-api-grammar")]},
    )
    assert opened.status_code == 200
    grammar = api_client.get("/api/profiles/es-es/grammar")
    assert grammar.status_code == 200
    assert grammar.json()["constructions"][0]["key"] == "es:present-indicative"


def test_event_api_idempotency_and_error_mappings(
    api_client: TestClient,
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    lesson = import_lesson(db, lesson_factory())
    reveal = event_factory(
        lesson.id,
        "term.revealed",
        event_id="api-reveal",
        payload={"term_key": "es:manana:NOUN"},
    )

    accepted = api_client.post("/api/profiles/es-es/events", json={"events": [reveal]})
    duplicate = api_client.post("/api/profiles/es-es/events", json={"events": [reveal]})
    assert accepted.status_code == 200
    assert accepted.json()["accepted"] == 1
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicates"] == 1

    conflict = deepcopy(reveal)
    conflict["payload"] = {"term_key": "es:corazon:NOUN"}
    response = api_client.post("/api/profiles/es-es/events", json={"events": [conflict]})
    assert response.status_code == 409

    missing_lesson = event_factory(
        999,
        "lesson.started",
        event_id="missing-lesson",
    )
    response = api_client.post("/api/profiles/es-es/events", json={"events": [missing_lesson]})
    assert response.status_code == 404

    unknown_term = event_factory(
        lesson.id,
        "term.revealed",
        event_id="unknown-term",
        payload={"term_key": "missing"},
    )
    response = api_client.post("/api/profiles/es-es/events", json={"events": [unknown_term]})
    assert response.status_code == 422


def test_reader_api_returns_lesson_and_maps_missing_id(
    api_client: TestClient, db: Session, lesson_factory: Any
) -> None:
    lesson = import_lesson(db, lesson_factory())

    response = api_client.get("/api/profiles/es-es/reader")
    assert response.status_code == 200
    assert response.json()["lesson_id"] == lesson.id
    assert response.json()["lesson"]["key"] == "lesson-one"

    missing = api_client.get("/api/profiles/es-es/reader", params={"lesson_id": 999})
    assert missing.status_code == 404


def test_public_base_path_prefixes_browser_urls(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import server.main

    monkeypatch.setattr(server.main, "BASE_PATH", "/lang")
    monkeypatch.setattr(server.main, "app_path", lambda path: f"/lang{path}")

    root = api_client.get("/", follow_redirects=False)
    page = api_client.get("/p/es-es", params={"view": "texts"})

    assert root.headers["location"].startswith("/lang/p/")
    assert 'data-base-path="/lang"' in page.text
    assert 'href="/lang/p/es-es?view=texts"' in page.text
    assert 'src="/lang/assets/app.js"' in page.text
    assert 'href="/p/' not in page.text
    assert "Path=/lang" in page.headers["set-cookie"]


def test_base_path_rejects_ambiguous_prefixes() -> None:
    from server.settings import parse_base_path

    assert parse_base_path("") == ""
    assert parse_base_path("/lang") == "/lang"
    for value in ("/", "lang", "/lang/", "/a b", "//x"):
        with pytest.raises(ValueError):
            parse_base_path(value)

"""Fingerprint of every profile input that changes what a generated lesson should be."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from server.agent_brief import calibration_generation_brief
from server.grammar_catalog import (
    find_grammar_catalog,
    grammar_catalog_digest,
)
from server.language_packs import language_pack_digest
from server.models import (
    Profile,
)
from server.profile_state import ensure_profile, profile_level_source


def profile_fingerprint(db: Session, profile: Profile | None = None) -> str:
    stored_profile = profile or ensure_profile(db)
    calibration = calibration_generation_brief(db)
    profile_payload: dict[str, Any] = {
        "learning_language": stored_profile.learning_language,
        "translation_language": stored_profile.translation_language,
        "level": stored_profile.level,
        "difficulty": stored_profile.difficulty,
        "interests": list(stored_profile.interests),
        "preferences": {
            key: value for key, value in stored_profile.preferences.items() if key != "tts"
        },
    }
    if profile_level_source(stored_profile) == "estimated":
        profile_payload.pop("difficulty")
        profile_payload.pop("level")
    serialized = {
        "profile": profile_payload,
        "calibration": calibration.model_dump(mode="json") if calibration is not None else None,
        "language_pack_digest": language_pack_digest(stored_profile.learning_language),
        "grammar_catalog_digest": (
            grammar_catalog_digest(stored_profile.learning_language)
            if find_grammar_catalog(stored_profile.learning_language) is not None
            else None
        ),
    }
    return hashlib.sha256(canonical_json(serialized).encode()).hexdigest()


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

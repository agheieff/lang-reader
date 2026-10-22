"""Explicit profile activation and optional onboarding seed data."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator
from sqlalchemy.orm import Session

from server.calibration import cefr_for_difficulty
from server.clock import as_utc, utc_now
from server.models import Profile
from server.schemas import StrictModel

ACTIVATION_PREFERENCE_KEY = "activation"
ONBOARDING_PREFERENCE_KEY = "onboarding"
PROFILE_LEVEL_SOURCE_KEY = "level_source"

StartingPoint = Literal[
    "unsure",
    "complete_beginner",
    "basic_phrases",
    "simple_texts",
    "general_reading",
    "complex_texts",
    "advanced_reading",
    "near_native",
]
QuestionnaireConfidence = Literal["low", "medium", "high"]

STARTING_POINT_DIFFICULTY: dict[StartingPoint, float | None] = {
    "unsure": None,
    "complete_beginner": 0.02,
    "basic_phrases": 0.10,
    "simple_texts": 0.25,
    "general_reading": 5 / 12,
    "complex_texts": 7 / 12,
    "advanced_reading": 0.75,
    "near_native": 0.94,
}
STARTING_POINT_SIGMA: dict[StartingPoint, float | None] = {
    "unsure": None,
    "complete_beginner": 0.10,
    "basic_phrases": 0.16,
    "simple_texts": 0.18,
    "general_reading": 0.18,
    "complex_texts": 0.18,
    "advanced_reading": 0.16,
    "near_native": 0.10,
}
QUESTIONNAIRE_SIGMA_FACTOR: dict[QuestionnaireConfidence, float] = {
    "low": 1.25,
    "medium": 1.0,
    "high": 0.8,
}


class ProfileActivationUpdate(StrictModel):
    starting_point: StartingPoint = "unsure"
    confidence: QuestionnaireConfidence = "medium"
    interests: list[str] = Field(default_factory=list, max_length=12)
    text_length: int | None = Field(default=None, ge=50, le=2_000)

    @field_validator("interests")
    @classmethod
    def normalize_interests(cls, value: list[str]) -> list[str]:
        normalized = [" ".join(item.split()) for item in value]
        normalized = [item for item in normalized if item]
        if any(len(item) > 100 for item in normalized):
            raise ValueError("interests must contain at most 100 characters each")
        return list(dict.fromkeys(normalized))


class ProfileActivationView(StrictModel):
    active: bool
    learning_language: str
    activated_at: datetime | None = None
    questionnaire_completed: bool
    starting_point: StartingPoint | None = None
    confidence: QuestionnaireConfidence | None = None


def profile_is_active(profile: Profile) -> bool:
    value = profile.preferences.get(ACTIVATION_PREFERENCE_KEY)
    if not isinstance(value, dict):
        # Existing profiles are migrated by ensure_profile(). Keep this fallback safe for callers
        # holding a row created by an older process during a rolling local restart.
        return True
    return value.get("active") is True


def activation_view(profile: Profile) -> ProfileActivationView:
    raw_activation = profile.preferences.get(ACTIVATION_PREFERENCE_KEY)
    activation = raw_activation if isinstance(raw_activation, dict) else {}
    raw_onboarding = profile.preferences.get(ONBOARDING_PREFERENCE_KEY)
    onboarding = raw_onboarding if isinstance(raw_onboarding, dict) else {}
    starting_point = onboarding.get("starting_point")
    confidence = onboarding.get("confidence")
    return ProfileActivationView(
        active=profile_is_active(profile),
        learning_language=profile.learning_language,
        activated_at=_optional_datetime(activation.get("activated_at")),
        questionnaire_completed=bool(onboarding),
        starting_point=starting_point if starting_point in STARTING_POINT_DIFFICULTY else None,
        confidence=confidence if confidence in QUESTIONNAIRE_SIGMA_FACTOR else None,
    )


def activate_profile_settings(
    db: Session, update: ProfileActivationUpdate
) -> ProfileActivationView:
    profile = db.get(Profile, 1)
    if profile is None:
        raise LookupError("profile is not configured")
    if profile_is_active(profile):
        return activation_view(profile)

    now = utc_now()
    preferences = dict(profile.preferences)
    seed = STARTING_POINT_DIFFICULTY[update.starting_point]
    base_sigma = STARTING_POINT_SIGMA[update.starting_point]
    preferences[ACTIVATION_PREFERENCE_KEY] = {
        "schema_version": 1,
        "active": True,
        "activated_at": now.isoformat(),
    }
    if update.starting_point != "unsure":
        preferences[ONBOARDING_PREFERENCE_KEY] = {
            "starting_point": update.starting_point,
            "confidence": update.confidence,
            "difficulty": seed,
            "variance": (base_sigma * QUESTIONNAIRE_SIGMA_FACTOR[update.confidence]) ** 2
            if base_sigma is not None
            else None,
            "answered_at": now.isoformat(),
        }
    if update.text_length is not None:
        preferences["text_length"] = update.text_length
    if update.interests:
        profile.interests = update.interests
    if seed is not None:
        profile.difficulty = seed
        profile.level = cefr_for_difficulty(seed)
        # A questionnaire answer is a revisable prior, even when an agent configured a dormant
        # placeholder level first. Calibration must remain allowed to replace it.
        preferences[PROFILE_LEVEL_SOURCE_KEY] = "unknown"
    profile.preferences = preferences
    db.commit()
    db.refresh(profile)
    from server.profile_state import rebuild_proficiency_state

    rebuild_proficiency_state(db)
    return activation_view(profile)


def questionnaire_seed(profile: Profile) -> tuple[float, float] | None:
    raw = profile.preferences.get(ONBOARDING_PREFERENCE_KEY)
    if not isinstance(raw, dict):
        return None
    difficulty = raw.get("difficulty")
    variance = raw.get("variance")
    if (
        isinstance(difficulty, bool)
        or not isinstance(difficulty, (int, float))
        or not 0 <= difficulty <= 1
        or isinstance(variance, bool)
        or not isinstance(variance, (int, float))
        or variance <= 0
    ):
        return None
    return float(difficulty), float(variance)


def _optional_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return as_utc(parsed)

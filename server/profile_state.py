"""Profile storage, level source, and the derived profile view."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.calibration import (
    INITIAL_PROBE_DIFFICULTY,
    cefr_center,
    cefr_for_difficulty,
)
from server.clock import as_utc, utc_now
from server.generation_tasks import (
    GenerationMode,
)
from server.models import (
    Interaction,
    Lesson,
    ProficiencyState,
    Profile,
)
from server.proficiency import rebuild_proficiency_state as _rebuild_proficiency_state
from server.profile_activation import (
    ACTIVATION_PREFERENCE_KEY,
    ONBOARDING_PREFERENCE_KEY,
    PROFILE_LEVEL_SOURCE_KEY,
)
from server.schemas import (
    CefrLevel,
    LevelSource,
    ProficiencyView,
    ProfileUpdate,
    ProfileView,
)
from server.workspaces import Workspace

PROFILE_LEVEL_SET_AT_KEY = "level_set_at"
PROFILE_LEVEL_EVIDENCE_CURSOR_KEY = "level_evidence_cursor"


class ProfileInactiveError(ValueError):
    pass


def ensure_profile(
    db: Session,
    *,
    learning_language: str = "es-ES",
    translation_language: str = "en",
    activated: bool | None = None,
) -> Profile:
    profile = db.get(Profile, 1)
    if profile is None:
        profile = Profile(
            id=1,
            learning_language=learning_language,
            translation_language=translation_language,
            level=cefr_for_difficulty(INITIAL_PROBE_DIFFICULTY),
            difficulty=INITIAL_PROBE_DIFFICULTY,
            preferences={
                PROFILE_LEVEL_SOURCE_KEY: "unknown",
                ACTIVATION_PREFERENCE_KEY: {
                    "schema_version": 1,
                    "active": True if activated is None else activated,
                    "activated_at": utc_now().isoformat() if activated else None,
                },
            },
        )
        db.add(profile)
    else:
        preferences = dict(profile.preferences)
        if PROFILE_LEVEL_SOURCE_KEY not in preferences:
            preferences[PROFILE_LEVEL_SOURCE_KEY] = "self_reported"
        if (
            preferences[PROFILE_LEVEL_SOURCE_KEY] == "self_reported"
            and PROFILE_LEVEL_SET_AT_KEY not in preferences
        ):
            preferences[PROFILE_LEVEL_SET_AT_KEY] = as_utc(profile.updated_at).isoformat()
        if (
            preferences[PROFILE_LEVEL_SOURCE_KEY] == "self_reported"
            and PROFILE_LEVEL_EVIDENCE_CURSOR_KEY not in preferences
        ):
            anchor = _stored_level_anchor(preferences, profile.updated_at)
            preferences[PROFILE_LEVEL_EVIDENCE_CURSOR_KEY] = (
                db.scalar(select(func.max(Interaction.id)).where(Interaction.occurred_at <= anchor))
                or 0
            )
        if ACTIVATION_PREFERENCE_KEY not in preferences:
            has_lessons = bool(db.scalar(select(func.count()).select_from(Lesson)))
            preferences[ACTIVATION_PREFERENCE_KEY] = {
                "schema_version": 1,
                "active": has_lessons,
                "activated_at": None,
                "migrated": True,
            }
        if preferences != profile.preferences:
            profile.preferences = preferences
    if db.get(ProficiencyState, 1) is None:
        db.add(ProficiencyState(id=1))
    if db.new or db.dirty:
        db.commit()
        db.refresh(profile)
    return profile


def update_profile(db: Session, patch: ProfileUpdate | Mapping[str, Any]) -> Profile:
    update = patch if isinstance(patch, ProfileUpdate) else ProfileUpdate.model_validate(patch)
    profile = ensure_profile(db)
    values = update.model_dump(exclude_none=True)
    level_changed = update.level is not None or update.difficulty is not None
    reserved = {
        key: profile.preferences[key]
        for key in (ACTIVATION_PREFERENCE_KEY, ONBOARDING_PREFERENCE_KEY)
        if key in profile.preferences
    }
    if level_changed:
        values["level"] = (
            update.level
            if update.level is not None
            else cefr_for_difficulty(cast(float, update.difficulty))
        )
        values["difficulty"] = (
            update.difficulty
            if update.difficulty is not None
            else cefr_center(cast(CefrLevel, update.level))
        )
        preferences = dict(profile.preferences)
        preferences.update(update.preferences or {})
        preferences.update(reserved)
        preferences[PROFILE_LEVEL_SOURCE_KEY] = "self_reported"
        preferences[PROFILE_LEVEL_SET_AT_KEY] = utc_now().isoformat()
        preferences[PROFILE_LEVEL_EVIDENCE_CURSOR_KEY] = (
            db.scalar(select(func.max(Interaction.id))) or 0
        )
        values["preferences"] = preferences
    elif update.preferences is not None:
        preferences = dict(profile.preferences)
        preferences.update(update.preferences)
        preferences.update(reserved)
        preferences[PROFILE_LEVEL_SOURCE_KEY] = profile_level_source(profile)
        values["preferences"] = preferences
    for key, value in values.items():
        setattr(profile, key, value)
    db.commit()
    db.refresh(profile)
    if level_changed:
        rebuild_proficiency_state(db)
        db.refresh(profile)
    return profile


def profile_level_source(profile: Profile) -> LevelSource:
    value = profile.preferences.get(PROFILE_LEVEL_SOURCE_KEY)
    if value not in {"unknown", "self_reported", "estimated"}:
        raise ValueError("profile has an invalid level source")
    return cast(LevelSource, value)


def generation_mode(profile: Profile) -> GenerationMode:
    return "calibration" if profile_level_source(profile) == "unknown" else "lesson"


def rebuild_proficiency_state(db: Session) -> ProficiencyState:
    """Rebuild proficiency while keeping profile migration and anchoring in this facade."""

    profile = ensure_profile(db)
    source = profile_level_source(profile)
    evidence_cursor = (
        _self_reported_level_evidence_cursor(profile) if source == "self_reported" else None
    )
    return _rebuild_proficiency_state(
        db,
        profile=profile,
        source=source,
        evidence_cursor=evidence_cursor,
    )


def build_profile_view(db: Session, profile: Profile | None = None) -> ProfileView:
    profile = profile or ensure_profile(db)
    state = db.get(ProficiencyState, 1)
    difficulty = (
        state.estimate if state is not None and state.estimate is not None else profile.difficulty
    )
    proficiency = ProficiencyView(
        source=profile_level_source(profile),
        status=state.status if state is not None else "unstarted",
        lower=state.lower if state is not None else None,
        upper=state.upper if state is not None else None,
        lower_level=(
            cefr_for_difficulty(state.lower)
            if state is not None and state.lower is not None
            else None
        ),
        upper_level=(
            cefr_for_difficulty(state.upper)
            if state is not None and state.upper is not None
            else None
        ),
        qualified_attempts=state.qualified_attempts if state is not None else 0,
        usable_probes=state.usable_probes if state is not None else 0,
        qualified_readings=state.qualified_readings if state is not None else 0,
    )
    return ProfileView(
        learning_language=profile.learning_language,
        translation_language=profile.translation_language,
        level=cefr_for_difficulty(difficulty),
        difficulty=difficulty,
        interests=list(profile.interests),
        preferences=dict(profile.preferences),
        proficiency=proficiency,
    )


def _stored_level_anchor(preferences: Mapping[str, Any], fallback: datetime) -> datetime:
    value = preferences.get(PROFILE_LEVEL_SET_AT_KEY)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            pass
        else:
            return as_utc(parsed)
    return as_utc(fallback)


def _self_reported_level_evidence_cursor(profile: Profile) -> int:
    value = profile.preferences.get(PROFILE_LEVEL_EVIDENCE_CURSOR_KEY)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("self-reported profile has an invalid level evidence cursor")
    return value


def profile_key(profile: Profile) -> str:
    raw = f"{profile.learning_language}--{profile.translation_language}".lower()
    parts = [part for part in "".join(char if char.isalnum() else " " for char in raw).split()]
    return "-".join(parts)


def workspace_key_and_path(db: Session, profile: Profile) -> tuple[str, str]:
    workspace = db.info.get("workspace")
    if isinstance(workspace, Workspace):
        return workspace.profile_id, workspace.relative_directory
    key = profile_key(profile)
    return key, f"profiles/{key}"

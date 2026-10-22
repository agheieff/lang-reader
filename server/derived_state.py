"""Keep the derived learning caches current without rebuilding on every request."""

from __future__ import annotations

import hashlib
import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from server.clock import utc_now
from server.grammar_catalog import find_grammar_catalog, grammar_catalog_digest
from server.language_packs import language_pack_digest
from server.models import DerivedState, Interaction, Lesson, Profile

# Bump when a reducer's semantics change so existing caches are replayed with the new code.
DERIVED_MODEL_REVISION = 2


def evidence_watermark(db: Session) -> str:
    """Summarize every input of the derived caches.

    Interactions are append-only, lessons change only through import, and the profile row changes
    on any setting update. The UTC date bounds staleness of time-weighted proficiency evidence.
    """

    interactions = db.execute(select(func.max(Interaction.id), func.count(Interaction.id))).one()
    lessons = db.execute(
        select(func.max(Lesson.id), func.count(Lesson.id), func.max(Lesson.updated_at))
    ).one()
    profile = db.get(Profile, 1)
    language = profile.learning_language if profile is not None else None
    parts = [
        DERIVED_MODEL_REVISION,
        list(interactions),
        [str(value) for value in lessons],
        str(profile.updated_at) if profile is not None else None,
        language_pack_digest(language) if language else None,
        grammar_catalog_digest(language)
        if language and find_grammar_catalog(language) is not None
        else None,
        utc_now().date().isoformat(),
    ]
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()


def ensure_derived_states(db: Session) -> bool:
    """Replay derived caches only when their evidence changed; return whether it rebuilt."""

    from server.character_learning import rebuild_character_states
    from server.grammar import rebuild_grammar_states
    from server.lexeme_learning import rebuild_lexeme_states
    from server.profile_state import rebuild_proficiency_state

    watermark = evidence_watermark(db)
    state = db.get(DerivedState, 1)
    if state is not None and state.watermark == watermark:
        return False
    rebuild_proficiency_state(db)  # first: the vocabulary prior uses the learner's frontier
    rebuild_lexeme_states(db)
    rebuild_character_states(db)
    rebuild_grammar_states(db)
    # Proficiency may update the profile's level source, so record the post-rebuild watermark.
    state = db.get(DerivedState, 1) or DerivedState(id=1)
    state.watermark = evidence_watermark(db)
    state.rebuilt_at = utc_now()
    db.add(state)
    db.commit()
    return True

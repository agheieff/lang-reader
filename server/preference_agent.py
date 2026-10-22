"""Let the agent maintain the reading-preference notes from learner messages and reactions."""

from __future__ import annotations

import json
import shutil
import time

from pydantic import ValidationError

from server.clock import utc_now
from server.db import session_scope
from server.generation_callback import (
    CallbackDeferred,
    CallbackInvocation,
    GenerationCallback,
    callback_result_error,
    canonical_request,
    strict_output_schema,
    write_private,
)
from server.profile_activation import profile_is_active
from server.profile_state import ensure_profile
from server.prompting import build_prompt
from server.reading_preferences import (
    apply_agent_update,
    content_history,
    current_preferences,
    current_revision,
    pending_messages,
    update_due,
)
from server.schemas import PreferenceUpdateRequest, PreferenceUpdateResult
from server.workspaces import Workspace

FAILURE_BACKOFF_SECONDS = 1_800.0
_last_failure: dict[str, float] = {}

PREFERENCE_UPDATE_INSTRUCTIONS = (
    "You maintain the learner's reading-preference notes, which guide the choice of subjects for "
    "their graded reading texts. Return an updated version of current_preferences as concise "
    "plain text with short sections such as Likes, Dislikes, Open questions / worth testing, and "
    "Style. Treat every user_message as authoritative intent: fold it in, and let it override "
    "earlier notes or inferences. Treat content_history as weaker evidence: liked and finished "
    "texts support a subject, disliked, skipped, or abandoned ones count against it, and one "
    "reaction is weak on its own. For texts with a hypothesis, record the answer the reaction "
    "suggests, tentatively unless repeated. Keep the learner's own wording where they wrote it, "
    "never drop an explicit statement unless a later message contradicts it, and stay under 2500 "
    "characters. Write likes as subjects and domains (content_history gives each text's "
    "domain), not as plot mechanics: if liked texts share a story shape, such as investigating "
    "a malfunction, record only the subjects as likes and list the shape under Open "
    "questions / worth testing as overused, unless the learner asked for it. Name domains the "
    "history shows as never or rarely tried under Open questions / worth testing. In reason, "
    "say in one sentence what changed and why."
)


def maybe_update_preferences(
    workspace: Workspace, callback: GenerationCallback, *, force: bool = False
) -> bool:
    """Run one agent update when due (or when forced); return whether a revision was stored.

    ``force`` skips only the "enough new reactions" gate and the failure backoff. The inputs are
    the same stored messages and reactions, so a forced update is the normal update, run early.
    """

    failed_at = _last_failure.get(workspace.profile_id)
    backed_off = failed_at is not None and time.monotonic() - failed_at < FAILURE_BACKOFF_SECONDS
    if backed_off and not force:
        return False
    with session_scope(workspace) as db:
        profile = ensure_profile(db)
        if not profile_is_active(profile) or not (force or update_due(db)):
            return False
        revision = current_revision(db)
        base_revision_id = revision.id if revision is not None else None
        messages = pending_messages(db)
        request = PreferenceUpdateRequest(
            job_id=f"{workspace.profile_id}-preferences",
            task_id=0,
            profile_key=workspace.profile_id,
            workspace_path=workspace.relative_directory,
            learning_language=profile.learning_language,
            current_preferences=current_preferences(db),
            user_messages=[message.text for message in messages][-20:],
            content_history=content_history(db),
            instructions=PREFERENCE_UPDATE_INSTRUCTIONS,
        )
        message_ids = [message.id for message in messages]

    job_dir = workspace.directory / "agent" / "preferences" / utc_now().strftime("%Y%m%dT%H%M%S")
    job_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    invocation = CallbackInvocation(
        request=request,
        job_dir=job_dir,
        request_path=job_dir / "request.json",
        response_schema_path=job_dir / "response-schema.json",
        response_path=job_dir / "response.json",
        prompt_path=job_dir / "prompt.txt",
        usage_path=job_dir / "usage.json",
    )
    write_private(invocation.request_path, canonical_request(request))
    write_private(job_dir / "prompt.txt", build_prompt(request))
    write_private(
        invocation.response_schema_path,
        json.dumps(strict_output_schema(PreferenceUpdateResult.model_json_schema())),
    )
    try:
        result = callback.run(invocation)
        error = callback_result_error(result)
        if error is not None:
            raise ValueError(error)
        update = PreferenceUpdateResult.model_validate_json(result.payload)
    except (OSError, ValueError, ValidationError, CallbackDeferred) as error:
        _last_failure[workspace.profile_id] = time.monotonic()
        print(f"preference update for {workspace.profile_id} failed: {error}", flush=True)
        return False
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    _last_failure.pop(workspace.profile_id, None)
    with session_scope(workspace) as db:
        # A learner edit made meanwhile wins; the next maintenance pass recomputes from it.
        return apply_agent_update(
            db,
            base_revision_id=base_revision_id,
            text=update.preferences,
            reason=update.reason,
            message_ids=message_ids,
        )

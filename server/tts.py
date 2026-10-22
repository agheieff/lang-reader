"""Provider-neutral lesson audio settings, caching, and durable work queue."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from server import settings
from server.audio_files import MEDIA_TYPES, configured_suffix
from server.base_path import app_path
from server.clock import utc_now
from server.lesson_content import body_sentences, lesson_document
from server.models import AudioTask, Interaction, Lesson, Profile
from server.profile_activation import profile_is_active
from server.schemas import LessonDocument, StrictModel
from server.workspaces import DATA_DIR, PROJECT_ROOT, Workspace

TTS_PROVIDER = "qwen3-tts"
TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
TTS_MODEL_REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
TTS_SEGMENTATION_REVISION = 1
TTS_PREFERENCE_KEY = "tts"
MAX_AUDIO_ATTEMPTS = 2

AudioTaskState = Literal["pending", "running", "completed", "failed", "superseded"]
AudioAvailability = Literal[
    "ready", "preparing", "failed", "unavailable", "unsupported", "disabled"
]
AudioStatusReason = Literal[
    "ready",
    "queued",
    "generating",
    "retrying",
    "generation_timeout",
    "invalid_audio",
    "provider_stopped",
    "generation_failed",
    "provider_unavailable",
    "unsupported_language",
    "profile_inactive",
]


@dataclass(frozen=True)
class VoiceSpec:
    id: str
    label: str
    description: str
    native_language: str


QWEN_VOICES = (
    VoiceSpec("Vivian", "Vivian", "Bright, clear young voice", "Chinese"),
    VoiceSpec("Serena", "Serena", "Warm, gentle young voice", "Chinese"),
    VoiceSpec("Uncle_Fu", "Uncle Fu", "Low, mellow mature voice", "Chinese"),
    VoiceSpec("Dylan", "Dylan", "Clear, youthful Beijing voice", "Chinese (Beijing)"),
    VoiceSpec("Eric", "Eric", "Lively, slightly husky Chengdu voice", "Chinese (Sichuan)"),
    VoiceSpec("Ryan", "Ryan", "Dynamic voice with strong rhythm", "English"),
    VoiceSpec("Aiden", "Aiden", "Sunny American voice", "English (US)"),
    VoiceSpec("Ono_Anna", "Ono Anna", "Light, playful voice", "Japanese"),
    VoiceSpec("Sohee", "Sohee", "Warm voice with a rich tone", "Korean"),
)
_VOICE_BY_ID = {voice.id: voice for voice in QWEN_VOICES}
_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


class TtsVoiceView(StrictModel):
    id: str
    label: str
    description: str
    native_language: str
    recommended: bool = False


class TtsSettingsView(StrictModel):
    provider: str
    model: str
    learning_language: str
    model_language: str | None
    supported: bool
    provider_installed: bool
    profile_active: bool
    selected_voice_id: str | None
    voices: list[TtsVoiceView]
    note: str | None = None


class TtsSettingsUpdate(StrictModel):
    voice_id: str = Field(min_length=1, max_length=40)

    @field_validator("voice_id")
    @classmethod
    def strip_voice(cls, value: str) -> str:
        return value.strip()


class LessonAudioStatus(StrictModel):
    lesson_id: int
    state: AudioAvailability
    reason: AudioStatusReason
    voice_id: str | None
    voice_label: str | None
    audio_url: str | None = None
    message: str | None = None


class AudioProviderRequest(StrictModel):
    request_id: str
    model_id: str
    model_revision: str
    language: str
    voice_id: str
    chunks: list[str] = Field(min_length=1)
    output_path: str


def model_language(language_tag: str) -> str | None:
    return _LANGUAGE_NAMES.get(language_tag.split("-", 1)[0].casefold())


def recommended_voice_id(language_tag: str) -> str:
    base = language_tag.split("-", 1)[0].casefold()
    return {
        "zh": "Serena",
        "en": "Aiden",
        # Qwen has no German-native fixed preset. Ryan is the least region-specific of its
        # two English-native presets and remains fully compatible with the model's German mode.
        "de": "Ryan",
        "ja": "Ono_Anna",
        "ko": "Sohee",
    }.get(base, "Aiden")


def selected_voice_id(profile: Profile) -> str | None:
    if model_language(profile.learning_language) is None:
        return None
    raw_settings = profile.preferences.get(TTS_PREFERENCE_KEY, {})
    if isinstance(raw_settings, dict):
        voice = raw_settings.get("voice")
        if isinstance(voice, str) and voice in _VOICE_BY_ID:
            return voice
    return recommended_voice_id(profile.learning_language)


def provider_command() -> tuple[str, ...] | None:
    configured = settings.load().tts.command
    if configured:
        return configured
    runtime = DATA_DIR / "tts" / "qwen"
    python = runtime / ".venv" / "bin" / "python"
    if not python.is_file() or not (runtime / ".ready").is_file():
        return None
    provider = PROJECT_ROOT / "extras" / "tts-qwen" / "qwen_provider.py"
    return (str(python), str(provider), "--runtime-dir", str(runtime))


def get_tts_settings(profile: Profile) -> TtsSettingsView:
    language = model_language(profile.learning_language)
    recommended = recommended_voice_id(profile.learning_language)
    voice_id = selected_voice_id(profile)
    note = _language_note(profile.learning_language) if language is not None else None
    return TtsSettingsView(
        provider=TTS_PROVIDER,
        model=TTS_MODEL_ID,
        learning_language=profile.learning_language,
        model_language=language,
        supported=language is not None,
        provider_installed=provider_command() is not None,
        profile_active=profile_is_active(profile),
        selected_voice_id=voice_id,
        voices=[
            TtsVoiceView(
                id=voice.id,
                label=voice.label,
                description=voice.description,
                native_language=voice.native_language,
                recommended=voice.id == recommended,
            )
            for voice in QWEN_VOICES
        ]
        if language is not None
        else [],
        note=note,
    )


def _language_note(language_tag: str) -> str | None:
    normalized = language_tag.casefold()
    if normalized.startswith("zh-") or normalized == "zh":
        return (
            "Qwen reads both simplified and traditional text as Mandarin. "
            "Dylan and Eric intentionally retain regional Chinese character."
        )
    if normalized.startswith("es-") or normalized == "es":
        return (
            "The current compact model speaks Spanish but does not guarantee a Spain or Latin "
            "American accent; the text variant remains profile-specific."
        )
    if normalized.startswith("de-") or normalized == "de":
        return (
            "The current compact model speaks German, but its fixed voices do not include a "
            "native-German preset; Ryan is the initial compatible voice and can be changed here."
        )
    return None


def set_tts_voice(db: Session, voice_id: str) -> Profile:
    if voice_id not in _VOICE_BY_ID:
        raise ValueError(f"unsupported TTS voice: {voice_id}")
    profile = _profile(db)
    if model_language(profile.learning_language) is None:
        raise ValueError(f"Qwen TTS does not support {profile.learning_language}")
    preferences = dict(profile.preferences)
    raw_settings = preferences.get(TTS_PREFERENCE_KEY, {})
    settings = dict(raw_settings) if isinstance(raw_settings, dict) else {}
    settings.update({"provider": TTS_PROVIDER, "voice": voice_id})
    preferences[TTS_PREFERENCE_KEY] = settings
    profile.preferences = preferences
    db.commit()
    db.refresh(profile)
    if profile_is_active(profile):
        _supersede_stale_pending_tasks(db, voice_id)
        ensure_audio_backfill(db)
    return profile


def ensure_audio_backfill(db: Session) -> list[AudioTask]:
    profile = _profile(db)
    if not profile_is_active(profile):
        return []
    voice_id = selected_voice_id(profile)
    if voice_id is None:
        return []
    _supersede_stale_pending_tasks(db, voice_id)
    lessons = db.scalars(
        select(Lesson)
        .where(
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
        )
        .order_by(Lesson.imported_at, Lesson.id)
    ).all()
    return [ensure_lesson_audio_task(db, lesson, voice_id=voice_id) for lesson in lessons]


def ensure_lesson_audio_task(
    db: Session, lesson: Lesson, *, voice_id: str | None = None
) -> AudioTask:
    profile = _profile(db)
    if not profile_is_active(profile):
        raise ValueError("activate this language before preparing lesson audio")
    selected = voice_id or selected_voice_id(profile)
    if selected is None or selected not in _VOICE_BY_ID:
        raise ValueError(f"Qwen TTS does not support {lesson.learning_language}")
    if (
        lesson.learning_language != profile.learning_language
        or lesson.translation_language != profile.translation_language
    ):
        raise ValueError("lesson languages must match the local profile")
    cache_key = audio_cache_key(lesson, selected)
    task = db.scalar(select(AudioTask).where(AudioTask.cache_key == cache_key))
    if task is None:
        task = AudioTask(
            lesson_id=lesson.id,
            cache_key=cache_key,
            provider=TTS_PROVIDER,
            model_revision=TTS_MODEL_REVISION,
            learning_language=lesson.learning_language,
            voice_id=selected,
            state="pending",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
    elif task.state == "superseded":
        task.state = "pending"
        task.finished_at = None
        task.error = None
        task.updated_at = utc_now()
        db.commit()
        db.refresh(task)
    return task


def audio_cache_key(lesson: Lesson, voice_id: str) -> str:
    material = json.dumps(
        {
            "lesson_key": lesson.key,
            "source_hash": _audio_source_hash(lesson),
            "language": lesson.learning_language,
            "model": TTS_MODEL_ID,
            "model_revision": TTS_MODEL_REVISION,
            "provider": TTS_PROVIDER,
            "segmentation_revision": TTS_SEGMENTATION_REVISION,
            "voice": voice_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode()).hexdigest()


def _supersede_stale_pending_tasks(db: Session, voice_id: str) -> None:
    stale = db.scalars(
        select(AudioTask).where(
            AudioTask.state == "pending",
            AudioTask.voice_id != voice_id,
        )
    ).all()
    if not stale:
        return
    now = utc_now()
    for task in stale:
        task.state = "superseded"
        task.finished_at = now
        task.updated_at = now
    db.commit()


def recover_audio_tasks(db: Session) -> int:
    tasks = db.scalars(select(AudioTask).where(AudioTask.state == "running")).all()
    now = utc_now()
    for task in tasks:
        task.state = "pending" if task.attempts < MAX_AUDIO_ATTEMPTS else "failed"
        task.updated_at = now
        if task.state == "failed":
            task.finished_at = now
            task.error = "audio worker stopped before the task finished"
    if tasks:
        db.commit()
    return len(tasks)


def claim_audio_task(db: Session) -> AudioTask | None:
    profile = _profile(db)
    if not profile_is_active(profile):
        return None
    ensure_audio_backfill(db)
    voice_id = selected_voice_id(profile)
    if voice_id is None:
        return None
    completed = (
        select(Interaction.id)
        .where(
            Interaction.lesson_id == Lesson.id,
            Interaction.event_type == "lesson.completed",
        )
        .exists()
    )
    task = db.scalar(
        select(AudioTask)
        .join(Lesson, Lesson.id == AudioTask.lesson_id)
        .where(
            AudioTask.state == "pending",
            AudioTask.learning_language == profile.learning_language,
            AudioTask.voice_id == voice_id,
            AudioTask.provider == TTS_PROVIDER,
            AudioTask.model_revision == TTS_MODEL_REVISION,
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
        )
        .order_by(completed, Lesson.imported_at, Lesson.id, AudioTask.id)
        .limit(1)
    )
    if task is None:
        return None
    task.state = "running"
    task.attempts += 1
    task.started_at = utc_now()
    task.finished_at = None
    task.error = None
    task.updated_at = task.started_at
    db.commit()
    db.refresh(task)
    return task


def complete_audio_task(db: Session, task_id: int, *, relative_path: str) -> AudioTask:
    task = _running_task(db, task_id)
    task.state = "completed"
    task.relative_path = relative_path
    task.error = None
    task.finished_at = utc_now()
    task.updated_at = task.finished_at
    db.commit()
    db.refresh(task)
    return task


def fail_audio_task(db: Session, task_id: int, *, error: str) -> AudioTask:
    task = _running_task(db, task_id)
    now = utc_now()
    task.state = "pending" if task.attempts < MAX_AUDIO_ATTEMPTS else "failed"
    task.error = error[:2000]
    task.finished_at = now if task.state == "failed" else None
    task.updated_at = now
    db.commit()
    db.refresh(task)
    return task


def retry_audio_task(db: Session, task_id: int) -> AudioTask:
    """Give a failed task a fresh attempt budget, e.g. after fixing the audio runtime."""

    task = db.get(AudioTask, task_id)
    if task is None:
        raise LookupError(f"audio task not found: {task_id}")
    if task.state != "failed":
        raise ValueError(f"audio task {task_id} is not failed")
    task.state = "pending"
    task.attempts = 0
    task.finished_at = None
    task.updated_at = utc_now()
    db.commit()
    db.refresh(task)
    return task


def release_audio_task(db: Session, task_id: int, *, reason: str) -> AudioTask:
    """Return a claimed task to the queue without charging the attempt (worker-side problem)."""

    task = _running_task(db, task_id)
    task.state = "pending"
    task.attempts = max(0, task.attempts - 1)
    task.error = reason[:2000]
    task.started_at = None
    task.updated_at = utc_now()
    db.commit()
    db.refresh(task)
    return task


def _running_task(db: Session, task_id: int) -> AudioTask:
    task = db.get(AudioTask, task_id)
    if task is None:
        raise LookupError(f"audio task not found: {task_id}")
    if task.state != "running":
        raise ValueError(f"audio task {task_id} is not running")
    return task


def audio_status(db: Session, workspace: Workspace, lesson_id: int) -> LessonAudioStatus:
    profile = _profile(db)
    lesson = _lesson(db, profile, lesson_id)
    language = model_language(lesson.learning_language)
    voice_id = selected_voice_id(profile)
    if language is None or voice_id is None:
        return LessonAudioStatus(
            lesson_id=lesson.id,
            state="unsupported",
            reason="unsupported_language",
            voice_id=None,
            voice_label=None,
            message=f"Local neural speech is not available for {lesson.learning_language}.",
        )
    if not profile_is_active(profile):
        voice = _VOICE_BY_ID[voice_id]
        return LessonAudioStatus(
            lesson_id=lesson.id,
            state="disabled",
            reason="profile_inactive",
            voice_id=voice_id,
            voice_label=voice.label,
            message="Activate this language before preparing or playing lesson audio.",
        )
    task = ensure_lesson_audio_task(db, lesson, voice_id=voice_id)
    voice = _VOICE_BY_ID[voice_id]
    if task.state == "completed" and task.relative_path:
        path = safe_audio_path(workspace, task.relative_path)
        if path.is_file():
            return LessonAudioStatus(
                lesson_id=lesson.id,
                state="ready",
                reason="ready",
                voice_id=voice_id,
                voice_label=voice.label,
                audio_url=app_path(
                    f"/api/profiles/{workspace.profile_id}/lessons/{lesson.id}/audio"
                ),
            )
        task.state = "pending"
        task.relative_path = None
        task.error = "cached audio file is missing"
        task.updated_at = utc_now()
        db.commit()
    if provider_command() is None and not remote_tts_enabled():
        return LessonAudioStatus(
            lesson_id=lesson.id,
            state="unavailable",
            reason="provider_unavailable",
            voice_id=voice_id,
            voice_label=voice.label,
            message="The local Qwen runtime is not installed; a browser voice can be used instead.",
        )
    if task.state == "failed":
        failed_reason = _failed_audio_reason(task.error)
        return LessonAudioStatus(
            lesson_id=lesson.id,
            state="failed",
            reason=failed_reason,
            voice_id=voice_id,
            voice_label=voice.label,
            message=_failed_audio_message(failed_reason),
        )
    if task.state == "running":
        preparing_reason: AudioStatusReason = "generating"
        message = "Local audio is being generated now. Long texts can take several minutes."
    elif task.attempts > 0:
        preparing_reason = "retrying"
        message = "The first local generation attempt failed; one automatic retry is queued."
    elif remote_tts_enabled():
        preparing_reason = "queued"
        message = (
            "Audio is queued and is prepared by the audio worker while it is on; a browser "
            "voice can be used meanwhile."
        )
    else:
        preparing_reason = "queued"
        message = "Local audio is queued; the audio worker prepares one text at a time."
    return LessonAudioStatus(
        lesson_id=lesson.id,
        state="preparing",
        reason=preparing_reason,
        voice_id=voice_id,
        voice_label=voice.label,
        message=message,
    )


def _failed_audio_reason(error: str | None) -> AudioStatusReason:
    detail = (error or "").casefold()
    if "timed out" in detail or "timeout" in detail:
        return "generation_timeout"
    if "invalid wav" in detail or "invalid audio" in detail or "usable audio" in detail:
        return "invalid_audio"
    if "empty" in detail and "wav" in detail:
        return "invalid_audio"
    if "unsupported wav" in detail:
        return "invalid_audio"
    if "stopped unexpectedly" in detail or "worker stopped" in detail:
        return "provider_stopped"
    return "generation_failed"


def _failed_audio_message(reason: AudioStatusReason) -> str:
    return {
        "generation_timeout": "Local generation timed out after two attempts.",
        "invalid_audio": "The local model returned an invalid audio file.",
        "provider_stopped": "The local audio process stopped twice before finishing this text.",
        "generation_failed": "Local audio generation failed after two attempts.",
    }.get(reason, "Local audio generation failed.")


def completed_audio_path(db: Session, workspace: Workspace, lesson_id: int) -> Path:
    profile = _profile(db)
    if not profile_is_active(profile):
        raise LookupError("lesson audio is disabled until this language is activated")
    lesson = _lesson(db, profile, lesson_id)
    voice_id = selected_voice_id(profile)
    if voice_id is None:
        raise LookupError("lesson audio is unsupported")
    task = db.scalar(
        select(AudioTask).where(AudioTask.cache_key == audio_cache_key(lesson, voice_id))
    )
    if task is None or task.state != "completed" or not task.relative_path:
        raise LookupError("lesson audio is not ready")
    path = safe_audio_path(workspace, task.relative_path)
    if not path.is_file():
        raise LookupError("lesson audio cache is missing")
    return path


def safe_audio_path(workspace: Workspace, relative_path: str) -> Path:
    root = workspace.directory.resolve()
    path = (root / relative_path).resolve()
    if path.parent != (root / "audio").resolve() or path.suffix not in MEDIA_TYPES:
        raise ValueError("invalid lesson audio path")
    return path


def audio_output_paths(workspace: Workspace, task: AudioTask) -> tuple[Path, Path, str]:
    directory = workspace.directory / "audio"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    relative = f"audio/{task.cache_key}{configured_suffix()}"
    final_path = safe_audio_path(workspace, relative)
    temporary = directory / f".{task.cache_key}.{os.getpid()}.tmp.wav"
    return temporary, final_path, relative


def provider_request(task: AudioTask, lesson: Lesson, output_path: Path) -> AudioProviderRequest:
    document = lesson_document(lesson)
    language = model_language(document.learning_language)
    if language is None:
        raise ValueError(f"Qwen TTS does not support {document.learning_language}")
    return AudioProviderRequest(
        request_id=f"audio-{task.id}",
        model_id=TTS_MODEL_ID,
        model_revision=TTS_MODEL_REVISION,
        language=language,
        voice_id=task.voice_id,
        chunks=lesson_audio_chunks(document),
        output_path=str(output_path),
    )


def lesson_audio_chunks(document: LessonDocument, *, maximum: int = 500) -> list[str]:
    chunks: list[str] = []
    if document.title_sentence is not None:
        chunks.extend(_run_chunks([run.text for run in document.title_sentence.runs], maximum))
    else:
        chunks.extend(_plain_chunks(document.title, maximum))
    for sentence in body_sentences(document):
        chunks.extend(_run_chunks([run.text for run in sentence.runs], maximum))
    if not chunks:
        raise ValueError("lesson has no speakable text")
    return chunks


def _audio_source_hash(lesson: Lesson) -> str:
    document = lesson_document(lesson)
    canonical = "\n".join(lesson_audio_chunks(document))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _run_chunks(runs: list[str], maximum: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for run in runs:
        normalized = _normalize_speech(run)
        if not normalized:
            continue
        if len(normalized) > maximum:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_plain_chunks(normalized, maximum))
            continue
        if current and len(current) + len(normalized) > maximum:
            chunks.append(current)
            current = normalized
        else:
            current += normalized
    if current:
        chunks.append(current)
    return chunks


def _plain_chunks(text: str, maximum: int) -> list[str]:
    normalized = _normalize_speech(text)
    chunks: list[str] = []
    while len(normalized) > maximum:
        split_at = _natural_split(normalized, maximum)
        chunks.append(normalized[:split_at].strip())
        normalized = normalized[split_at:].strip()
    if normalized:
        chunks.append(normalized)
    return chunks


def _normalize_speech(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\x00", " "))


def _natural_split(text: str, maximum: int) -> int:
    floor = maximum // 2
    punctuation = "。！？!?；;，,、. "
    candidates = [text.rfind(mark, floor, maximum + 1) for mark in punctuation]
    split_at = max(candidates, default=-1)
    return split_at + 1 if split_at >= floor else maximum


def lesson_for_audio_task(db: Session, task: AudioTask) -> Lesson:
    lesson = db.get(Lesson, task.lesson_id)
    if lesson is None:
        raise LookupError(f"lesson not found: {task.lesson_id}")
    if audio_cache_key(lesson, task.voice_id) != task.cache_key:
        raise ValueError("audio task no longer matches the lesson content")
    return lesson


def _profile(db: Session) -> Profile:
    profile = db.get(Profile, 1)
    if profile is None:
        raise LookupError("profile not found")
    return profile


def _lesson(db: Session, profile: Profile, lesson_id: int) -> Lesson:
    lesson = db.scalar(
        select(Lesson).where(
            Lesson.id == lesson_id,
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
        )
    )
    if lesson is None:
        raise LookupError(f"lesson not found: {lesson_id}")
    return lesson


def project_environment() -> dict[str, str]:
    environment = dict(os.environ)
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{current}" if current else str(PROJECT_ROOT)
    )
    return environment


def audio_task_state(task: AudioTask) -> AudioTaskState:
    return cast(AudioTaskState, task.state)


def remote_tts_enabled() -> bool:
    """Audio is synthesized by a worker on another machine that claims tasks here."""

    return settings.load().tts.remote

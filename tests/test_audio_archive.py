from __future__ import annotations

import io
import shutil
import wave
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from server.audio_archive import compress_profile_audio
from server.db import init_db, session_scope
from server.learning import import_lesson
from server.models import AudioTask
from server.profile_activation import ProfileActivationUpdate, activate_profile_settings
from server.tts import audio_status, claim_audio_task, complete_audio_task
from server.workspaces import Workspace, WorkspaceRegistry

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def _wave(seconds: float = 2.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x01\x02" * int(24_000 * seconds))
    return buffer.getvalue()


@pytest.fixture
def workspace_with_wav(tmp_path: Path, lesson_factory: Any) -> Workspace:
    workspace = WorkspaceRegistry(tmp_path / "data").resolve("es-es")
    init_db(workspace)
    with session_scope(workspace) as db:
        activate_profile_settings(db, ProfileActivationUpdate())
        import_lesson(db, lesson_factory(key="archive-one"))
        audio_status(db, workspace, 1)  # enqueue the audio task
        task = claim_audio_task(db)
        assert task is not None
        relative = f"audio/{task.cache_key}.wav"
        (workspace.directory / "audio").mkdir(parents=True)
        (workspace.directory / relative).write_bytes(_wave())
        complete_audio_task(db, task.id, relative_path=relative)
    return workspace


def test_compress_moves_originals_aside_and_repoints_the_database(
    workspace_with_wav: Workspace, tmp_path: Path
) -> None:
    archive = tmp_path / "originals"

    report = compress_profile_audio(workspace_with_wav, archive, ".opus")

    assert report["compressed"] == 1 and report["bytes_after"] < report["bytes_before"] // 4
    with session_scope(workspace_with_wav) as db:
        path = db.scalars(select(AudioTask.relative_path)).one()
    assert path is not None and path.endswith(".opus")
    assert (workspace_with_wav.directory / path).is_file()
    assert not list((workspace_with_wav.directory / "audio").glob("*.wav"))
    originals = list((archive / "es-es").glob("*.wav"))
    assert len(originals) == 1 and originals[0].read_bytes() == _wave()


def test_compress_is_resumable_and_idempotent(
    workspace_with_wav: Workspace, tmp_path: Path
) -> None:
    archive = tmp_path / "originals"
    compress_profile_audio(workspace_with_wav, archive, ".opus")
    again = compress_profile_audio(workspace_with_wav, archive, ".opus")
    assert again["compressed"] == 0
    assert len(list((archive / "es-es").glob("*.wav"))) == 1


def test_compress_refuses_an_archive_inside_the_data_directory_and_wav_as_target(
    workspace_with_wav: Workspace,
) -> None:
    with pytest.raises(ValueError, match="outside"):
        compress_profile_audio(workspace_with_wav, workspace_with_wav.directory / "x", ".opus")
    with pytest.raises(ValueError, match="compressed format"):
        compress_profile_audio(workspace_with_wav, Path("/tmp/never-used-archive"), ".wav")

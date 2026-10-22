"""Compress a profile's existing WAV cache and keep the originals outside the data directory."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import select

from server.audio_files import require_encoder, transcode, validate_wave
from server.db import session_scope
from server.models import AudioTask
from server.tts import safe_audio_path
from server.workspaces import Workspace


def compress_profile_audio(workspace: Workspace, archive_dir: Path, suffix: str) -> dict[str, Any]:
    """Re-encode every completed WAV as ``suffix``; originals are moved, never deleted.

    Each file is encoded, decoded back and compared by duration before its database row
    changes; the original then moves to ``archive_dir/<profile>/``. Re-running resumes.
    """

    if suffix == ".wav":
        raise ValueError("choose a compressed format (.opus or .mp3)")
    require_encoder(suffix)
    archive = (archive_dir / workspace.profile_id).resolve()
    if archive.is_relative_to(workspace.directory.resolve()):
        raise ValueError("the archive must be outside the profile's data directory")
    archive.mkdir(parents=True, exist_ok=True, mode=0o700)
    compressed = bytes_before = bytes_after = 0
    with session_scope(workspace) as db:
        tasks = db.scalars(
            select(AudioTask).where(
                AudioTask.state == "completed", AudioTask.relative_path.like("%.wav")
            )
        ).all()
        for task in tasks:
            assert task.relative_path is not None
            source = safe_audio_path(workspace, task.relative_path)
            if not source.is_file():
                continue
            target = source.with_suffix(suffix)
            validate_wave(source)
            if not target.is_file():
                partial = target.with_name(f".{target.name}.part{suffix}")
                try:
                    transcode(source, partial)
                    partial.chmod(0o600)
                    partial.replace(target)
                finally:
                    partial.unlink(missing_ok=True)
            task.relative_path = f"audio/{target.name}"
            db.commit()
            bytes_before += source.stat().st_size
            bytes_after += target.stat().st_size
            shutil.move(str(source), archive / source.name)
            compressed += 1
    return {
        "profile": workspace.profile_id,
        "compressed": compressed,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "originals_in": str(archive),
    }

"""Cached lesson audio files: validation, compressed storage and serving types.

Synthesis providers produce WAV. The owner validates it and publishes it in the configured
format (``ARC_LANG_AUDIO_FORMAT``): ``wav`` unchanged, ``opus`` (Ogg Opus, about a tenth of the
size, needs ffmpeg with libopus) or ``mp3``. Existing files keep the format they were
published in; the database names each file's path.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import wave
from pathlib import Path

from server import settings

MEDIA_TYPES = {".wav": "audio/wav", ".opus": "audio/ogg; codecs=opus", ".mp3": "audio/mpeg"}
FFMPEG_ARGUMENTS = {
    ".opus": ["-c:a", "libopus", "-b:a", "32k", "-application", "audio", "-ac", "1"],
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "64k", "-ac", "1"],
}
FFMPEG_TIMEOUT_SECONDS = 600
DURATION_TOLERANCE_SECONDS = 0.5


def configured_suffix() -> str:
    return f".{settings.load().tts.audio_format}"


def media_type(path: Path) -> str:
    return MEDIA_TYPES[path.suffix]


def require_encoder(suffix: str) -> None:
    """Fail at worker start, not after a synthesis, when the chosen format cannot be written."""

    if suffix != ".wav" and shutil.which("ffmpeg") is None:
        raise ValueError(f"ARC_LANG_AUDIO_FORMAT={suffix[1:]} needs ffmpeg on the audio owner")


def validate_wave(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 44:
        raise ValueError("Qwen provider did not create usable audio")
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnframes() <= 0 or audio.getframerate() < 8_000:
                raise ValueError("Qwen provider created an empty or invalid WAV")
            if audio.getnchannels() not in {1, 2} or audio.getsampwidth() not in {2, 3, 4}:
                raise ValueError("Qwen provider created an unsupported WAV format")
            remaining = audio.getnframes()
            frame_bytes = audio.getnchannels() * audio.getsampwidth()
            while remaining:
                count = min(remaining, 65_536)
                if len(audio.readframes(count)) != count * frame_bytes:
                    raise ValueError("Qwen provider created a truncated WAV")
                remaining -= count
    except (wave.Error, EOFError) as error:
        raise ValueError("Qwen provider created an invalid WAV") from error


def wave_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def publish(validated_wav: Path, final_path: Path) -> None:
    """Move a validated WAV into the cache in the format named by ``final_path``'s suffix."""

    if final_path.suffix == ".wav":
        os.replace(validated_wav, final_path)
        final_path.chmod(0o600)
        return
    encoded = final_path.with_name(f".{final_path.name}.{os.getpid()}.part{final_path.suffix}")
    try:
        transcode(validated_wav, encoded)
        encoded.chmod(0o600)
        os.replace(encoded, final_path)
    finally:
        encoded.unlink(missing_ok=True)
        validated_wav.unlink(missing_ok=True)


def transcode(wav: Path, target: Path) -> None:
    """Encode ``wav`` to ``target`` and check the result decodes to the same duration."""

    arguments = FFMPEG_ARGUMENTS[target.suffix]
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), *arguments, str(target)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
        decoded = _decoded_seconds(target)
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", b"") or b""
        raise ValueError(
            f"audio encoding failed: {error} {detail.decode(errors='replace')}"
        ) from error
    if abs(decoded - wave_seconds(wav)) > DURATION_TOLERANCE_SECONDS:
        raise ValueError(f"encoded audio duration {decoded:.2f}s differs from the source")


def _decoded_seconds(path: Path) -> float:
    """Duration by decoding the whole file, which also proves it is intact."""

    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-", "-progress", "-"],
        check=True,
        capture_output=True,
        text=True,
        timeout=FFMPEG_TIMEOUT_SECONDS,
    )
    microseconds = [
        int(line.partition("=")[2])
        for line in completed.stdout.splitlines()
        if line.startswith("out_time_us=") and line.partition("=")[2].isdigit()
    ]
    if not microseconds:
        raise ValueError("could not measure the encoded audio")
    return microseconds[-1] / 1_000_000


def upload_digest(audio: bytes) -> str:
    return hashlib.sha256(audio).hexdigest()

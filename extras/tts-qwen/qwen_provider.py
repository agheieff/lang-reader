"""Isolated JSON-lines provider for Qwen3-TTS lesson audio.

This module runs inside the optional, heavyweight TTS environment created by ``install.sh``.
The core app talks to it over stdin/stdout so PyTorch and model dependencies never enter the
normal application environment. ``ARC_LANG_TTS_ATTENTION`` and ``ARC_LANG_TTS_DTYPE`` tune the
GPU run (see README.md in this directory).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, TextIO

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
MODEL_REVISION = "85e237c12c027371202489a0ec509ded67b5e4b5"
LANGUAGES = frozenset(
    {
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    }
)
VOICES = frozenset(
    {
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ryan",
        "Aiden",
        "Ono_Anna",
        "Sohee",
    }
)


class QwenProvider:
    def __init__(self, runtime_dir: Path) -> None:
        self._model: Any = None
        self._runtime_dir = runtime_dir

    def load(self) -> None:
        if self._model is not None:
            return
        with redirect_stdout(sys.stderr):
            import torch  # type: ignore[import-not-found]
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
            from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]

            device_file = self._runtime_dir / "device"
            device = (
                device_file.read_text(encoding="utf-8").strip() if device_file.is_file() else "cuda"
            )
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "Qwen GPU runtime is unavailable; the reader keeps its browser voice fallback"
                )
            ready_marker = self._runtime_dir / ".ready"
            snapshot = snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=ready_marker.is_file(),
            )
            # Eager FP32 is the conservative cross-driver default. This host's RDNA4 ROCm stack
            # loads FP16/SDPA successfully but raises a hardware exception during generation.
            attention = os.getenv("ARC_LANG_TTS_ATTENTION", "eager")
            if attention not in {"sdpa", "eager"}:
                raise ValueError("ARC_LANG_TTS_ATTENTION must be 'sdpa' or 'eager'")
            dtype_name = os.getenv("ARC_LANG_TTS_DTYPE", "float32")
            dtypes = {"float16": torch.float16, "float32": torch.float32}
            if dtype_name not in dtypes:
                raise ValueError("ARC_LANG_TTS_DTYPE must be 'float16' or 'float32'")
            self._model = Qwen3TTSModel.from_pretrained(
                snapshot,
                device_map="cuda:0" if device == "cuda" else "cpu",
                dtype=dtypes[dtype_name],
                attn_implementation=attention,
            )

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        _validate_request(request)
        self.load()
        import numpy as np  # type: ignore[import-not-found]
        import soundfile as sf  # type: ignore[import-not-found]

        language = str(request["language"])
        voice = str(request["voice_id"])
        chunks = request["chunks"]
        output = Path(str(request["output_path"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        silence: Any = None
        sample_rate: int | None = None
        pieces: list[Any] = []
        with redirect_stdout(sys.stderr):
            for chunk in chunks:
                wavs, rate = self._model.generate_custom_voice(
                    text=str(chunk),
                    language=language,
                    speaker=voice,
                )
                if not wavs:
                    raise RuntimeError("Qwen returned no audio")
                if sample_rate is None:
                    sample_rate = int(rate)
                    silence = np.zeros(round(sample_rate * 0.24), dtype=np.float32)
                elif int(rate) != sample_rate:
                    raise RuntimeError("Qwen changed sample rate within one lesson")
                audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
                if audio.size == 0:
                    raise RuntimeError("Qwen returned an empty audio chunk")
                if not np.isfinite(audio).all():
                    raise RuntimeError("Qwen returned non-finite audio samples")
                if audio.size > sample_rate * 300:
                    raise RuntimeError("Qwen returned an unexpectedly long audio chunk")
                if pieces:
                    pieces.append(silence)
                pieces.append(audio)
            assert sample_rate is not None
            joined = np.concatenate(pieces)
            if joined.size > sample_rate * 7_200:
                raise RuntimeError("Qwen returned lesson audio longer than two hours")
            sf.write(output, joined, sample_rate, subtype="PCM_16", format="WAV")
        return {
            "ok": True,
            "request_id": request["request_id"],
            "sample_rate": sample_rate,
            "samples": int(joined.size),
        }


def _validate_request(request: dict[str, Any]) -> None:
    if request.get("model_id") != MODEL_ID or request.get("model_revision") != MODEL_REVISION:
        raise ValueError("provider request uses an unsupported model revision")
    if request.get("language") not in LANGUAGES:
        raise ValueError("provider request uses an unsupported language")
    if request.get("voice_id") not in VOICES:
        raise ValueError("provider request uses an unsupported voice")
    chunks = request.get("chunks")
    if (
        not isinstance(chunks, list)
        or not chunks
        or len(chunks) > 200
        or any(
            not isinstance(chunk, str) or not chunk.strip() or len(chunk) > 500 for chunk in chunks
        )
    ):
        raise ValueError("provider request has invalid text chunks")
    output = request.get("output_path")
    if not isinstance(output, str) or not output.endswith(".tmp.wav"):
        raise ValueError("provider request has an invalid output path")


def _respond(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def serve(runtime_dir: Path) -> None:
    protocol = sys.stdout
    provider = QwenProvider(runtime_dir)
    for line in sys.stdin:
        value: Any = None
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("provider request must be an object")
            result = provider.generate(value)
        except Exception as error:
            result = {
                "ok": False,
                "request_id": value.get("request_id") if isinstance(value, dict) else None,
                "error": f"{type(error).__name__}: {error}",
            }
        _respond(protocol, result)


def warmup(runtime_dir: Path) -> None:
    provider = QwenProvider(runtime_dir)
    output = runtime_dir / ".install-smoke.tmp.wav"
    output.unlink(missing_ok=True)
    try:
        result = provider.generate(
            {
                "request_id": "install-smoke",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "language": "Chinese",
                "voice_id": "Serena",
                "chunks": ["你好，这是语音测试。"],
                "output_path": str(output),
            }
        )
    finally:
        output.unlink(missing_ok=True)
    _respond(
        sys.stdout,
        {
            "ok": True,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "languages": sorted(LANGUAGES),
            "voices": sorted(VOICES),
            "smoke_sample_rate": result["sample_rate"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated Qwen TTS provider")
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--warmup", action="store_true", help="download and load the pinned model")
    args = parser.parse_args()
    if args.warmup:
        warmup(args.runtime_dir)
    else:
        serve(args.runtime_dir)


if __name__ == "__main__":
    main()

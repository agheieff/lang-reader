#!/usr/bin/env bash
# Install the optional Qwen3-TTS runtime into <data dir>/tts/qwen (isolated from the app).
# Usage: extras/tts-qwen/install.sh rocm|cuda|cpu
set -euo pipefail

backend="${1:-}"
case "$backend" in
  rocm | cuda | cpu) ;;
  *)
    echo "usage: $0 rocm|cuda|cpu" >&2
    echo "  rocm  AMD GPU, pinned ROCm 7.2.1 wheels (Python 3.12, x86_64)" >&2
    echo "  cuda  NVIDIA GPU, PyTorch's default CUDA wheels" >&2
    echo "  cpu   works but is very slow: minutes of CPU per lesson" >&2
    exit 2
    ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
if [[ "$backend" == rocm ]]; then
  command -v rocminfo >/dev/null || { echo "rocminfo not found: is ROCm installed?" >&2; exit 1; }
  [[ "$(rocminfo 2>/dev/null)" =~ Name:[[:space:]]+gfx ]] || { echo "No ROCm GPU found" >&2; exit 1; }
fi

umask 077
DATA="${ARC_LANG_DATA_DIR:-data}"
RUNTIME="$DATA/tts/qwen"
PYTHON="$RUNTIME/.venv/bin/python"
mkdir -p "$RUNTIME"
uv venv --python 3.12 "$RUNTIME/.venv"

case "$backend" in
  rocm)
    BASE="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1"
    uv pip install --python "$PYTHON" "numpy==1.26.4" \
      "$BASE/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl" \
      "$BASE/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" \
      "$BASE/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"
    echo cuda > "$RUNTIME/device"  # PyTorch's ROCm build answers to the cuda device name
    ;;
  cuda)
    uv pip install --python "$PYTHON" "numpy==1.26.4" torch torchaudio
    echo cuda > "$RUNTIME/device"
    ;;
  cpu)
    uv pip install --python "$PYTHON" "numpy==1.26.4" torch torchaudio \
      --index-url https://download.pytorch.org/whl/cpu
    echo cpu > "$RUNTIME/device"
    ;;
esac
uv pip install --python "$PYTHON" "numpy==1.26.4" "qwen-tts==0.1.1"

rm -f "$RUNTIME/.ready"
"$PYTHON" extras/tts-qwen/qwen_provider.py --runtime-dir "$RUNTIME" --warmup
touch "$RUNTIME/.ready"
chmod 600 "$RUNTIME/.ready"
echo "Qwen3-TTS ($backend) is installed and its pinned model is ready. Restart the audio worker."

#!/usr/bin/env bash
set -euo pipefail

PORT=8000

while (($#)); do
  case "$1" in
    --port|-p)
      PORT="${2:?missing port}"
      shift 2
      ;;
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    *)
      echo "Usage: $0 [--port N]" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "${BASH_SOURCE[0]}")"
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
command -v pnpm >/dev/null || { echo "pnpm is required (Arch: sudo pacman -S pnpm)" >&2; exit 1; }

uv sync --locked
pnpm install --frozen-lockfile
pnpm build

WORKER_PIDS=()
WORKER_PGIDS=()
cleanup() {
  local exit_status=$?
  trap - EXIT INT TERM
  if ((${#WORKER_PIDS[@]})); then
    for pgid in "${WORKER_PGIDS[@]}"; do
      kill -TERM -- "-$pgid" 2>/dev/null || true
    done
    ( sleep 5; for pgid in "${WORKER_PGIDS[@]}"; do
        kill -KILL -- "-$pgid" 2>/dev/null || true
      done ) &
    local killer_pid=$!
    for pid in "${WORKER_PIDS[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
    kill "$killer_pid" 2>/dev/null || true
    wait "$killer_pid" 2>/dev/null || true
    for pgid in "${WORKER_PGIDS[@]}"; do
      kill -KILL -- "-$pgid" 2>/dev/null || true
    done
  fi
  exit "$exit_status"
}
trap cleanup EXIT INT TERM

start_worker() {
  setsid --wait "$@" &
  WORKER_PIDS+=("$!")
  WORKER_PGIDS+=("$!")
}

export ARC_LANG_AUTO_GENERATE="${ARC_LANG_AUTO_GENERATE:-1}"
if [[ "$ARC_LANG_AUTO_GENERATE" == "1" ]]; then
  command -v setsid >/dev/null || { echo "setsid is required (Arch: util-linux)" >&2; exit 1; }
  start_worker uv run python -m server.worker_supervisor
fi

if [[ "${ARC_LANG_TTS_ENABLED:-1}" == "1" && -x "${ARC_LANG_DATA_DIR:-data}/tts/qwen/.venv/bin/python" \
      && -f "${ARC_LANG_DATA_DIR:-data}/tts/qwen/.ready" ]]; then
  command -v setsid >/dev/null || { echo "setsid is required (Arch: util-linux)" >&2; exit 1; }
  start_worker uv run python -m server.tts_worker
fi

uv run uvicorn server.main:app --host 127.0.0.1 --port "$PORT" --reload \
  --reload-dir server --reload-include '*.py' --reload-include '*.toml' \
  --reload-exclude 'server/static/**'

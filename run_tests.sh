#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
uv run ruff check server tests scripts
uv run ruff format --check server tests scripts
uv run mypy server
uv run pytest
pnpm check
pnpm test
pnpm build
uv run python scripts/smoke_browser.py --if-available

from __future__ import annotations

from server import settings

# A reverse proxy may strip a fixed prefix such as `/lang` before forwarding.
# Routes stay unprefixed; only URLs sent back to the browser need the public prefix.
BASE_PATH = settings.load().base_path


def app_path(path: str) -> str:
    if not path.startswith("/"):
        raise ValueError("application paths must be absolute")
    return f"{BASE_PATH}{path}"

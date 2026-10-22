"""The single place that reads configuration from the environment.

Every ``ARC_LANG_*`` variable the application understands is parsed here into typed,
validated values, so a bad value fails at start instead of deep inside a request.
Variables that Lang sets for its own callback children (``ARC_LANG_PROTOCOL``,
``ARC_LANG_CALLBACK_*`` ...) are an output protocol, not settings, and live in
``agent_command.py``.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALLBACK_STAGES = ("prose", "lexical", "translation", "grammar", "preferences", "legacy")
AuthMode = Literal["none", "password", "proxy"]
_AUTH_MODES = ("none", "password", "proxy")


@dataclass(frozen=True)
class AgentSettings:
    """How generation work is scheduled; which model serves it is the provider file."""

    config_path: Path  # the provider file (providers.py)
    unavailable_exit_code: int
    poll_seconds: int
    maintenance_seconds: int
    lexical_batch_size: int | None
    admission_command: str | None


@dataclass(frozen=True)
class TtsSettings:
    command: tuple[str, ...] | None
    remote: bool
    timeout_seconds: float
    remote_host: str | None
    remote_command: str | None
    audio_format: Literal["wav", "opus"]
    worker_token: str | None  # owner side: enables /api/audio-worker/ for this bearer token
    remote_url: str | None  # worker side: the owner's base URL, used instead of ssh
    remote_token: str | None


@dataclass(frozen=True)
class AuthSettings:
    """Who may use the web interface. One learner per instance, never multi-user."""

    mode: AuthMode
    password_hash: str | None
    proxy_header: str
    proxy_addresses: tuple[str, ...]
    allowed_hosts: tuple[str, ...]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    base_path: str
    snapshot_dir: Path | None
    profile: str | None
    profile_cookie: str
    pack_dirs: tuple[Path, ...]
    auto_generate: bool
    queue_target: int
    agent: AgentSettings
    tts: TtsSettings
    auth: AuthSettings


def _integer(
    env: Mapping[str, str], name: str, default: int | None, low: int, high: int
) -> int | None:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _required_integer(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    value = _integer(env, name, default, low, high)
    assert value is not None
    return value


def _command(env: Mapping[str, str], name: str) -> tuple[str, ...] | None:
    raw = env.get(name)
    if not raw:
        return None
    argv = tuple(shlex.split(raw))
    if not argv:
        raise ValueError(f"{name} must not be empty")
    return argv


def _secret(env: Mapping[str, str], name: str) -> str | None:
    """A secret from ``NAME`` or from the file named by ``NAME_FILE``."""
    path = env.get(f"{name}_FILE")
    value = Path(path).read_text(encoding="utf-8").strip() if path else env.get(name, "")
    return value or None


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be 0 or 1")


def _list(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in env.get(name, "").split(",") if item.strip())


def parse_base_path(value: str) -> str:
    if not re.fullmatch(r"(/[A-Za-z0-9._~-]+)*", value):
        raise ValueError("ARC_LANG_BASE_PATH must be empty or like /lang, without a trailing slash")
    return value


def load(env: Mapping[str, str] | None = None) -> Settings:
    """Parse and validate the environment; raises ValueError naming the bad variable."""

    env = os.environ if env is None else env
    tts_timeout = _integer(env, "ARC_LANG_TTS_TIMEOUT_SECONDS", None, 60, 7200)
    audio_format = env.get("ARC_LANG_AUDIO_FORMAT", "wav")
    if audio_format not in {"wav", "opus"}:
        raise ValueError("ARC_LANG_AUDIO_FORMAT must be wav or opus")
    auth_mode = env.get("ARC_LANG_AUTH", "none")
    if auth_mode not in _AUTH_MODES:
        raise ValueError(f"ARC_LANG_AUTH must be one of {', '.join(_AUTH_MODES)}")
    password_hash = env.get("ARC_LANG_AUTH_PASSWORD_HASH") or None
    proxy_addresses = _list(env, "ARC_LANG_AUTH_PROXY_ADDRESSES")
    if auth_mode == "password" and not password_hash:
        raise ValueError("ARC_LANG_AUTH=password requires ARC_LANG_AUTH_PASSWORD_HASH")
    if auth_mode == "proxy" and not proxy_addresses:
        raise ValueError("ARC_LANG_AUTH=proxy requires ARC_LANG_AUTH_PROXY_ADDRESSES")
    snapshot = env.get("ARC_LANG_SNAPSHOT_DIR")
    return Settings(
        data_dir=Path(env.get("ARC_LANG_DATA_DIR") or PROJECT_ROOT / "data"),
        base_path=parse_base_path(env.get("ARC_LANG_BASE_PATH", "")),
        snapshot_dir=Path(snapshot) if snapshot else None,
        profile=env.get("ARC_LANG_PROFILE") or None,
        profile_cookie=env.get("ARC_LANG_PROFILE_COOKIE", "lang_profile"),
        pack_dirs=tuple(Path(item) for item in _list(env, "ARC_LANG_PACK_DIRS")),
        auto_generate=_flag(env, "ARC_LANG_AUTO_GENERATE", True),
        queue_target=_required_integer(env, "ARC_LANG_QUEUE_TARGET", 3, 1, 10),
        agent=AgentSettings(
            config_path=Path(env.get("ARC_LANG_CONFIG") or PROJECT_ROOT / "lang-reader.toml"),
            unavailable_exit_code=_required_integer(
                env, "ARC_LANG_AGENT_UNAVAILABLE_EXIT", 3, 1, 255
            ),
            poll_seconds=_required_integer(env, "ARC_LANG_AGENT_POLL_SECONDS", 2, 1, 60),
            maintenance_seconds=_required_integer(
                env, "ARC_LANG_AGENT_MAINTENANCE_SECONDS", 30, 1, 3600
            ),
            lexical_batch_size=_integer(env, "ARC_LANG_LEXICAL_BATCH_SIZE", None, 1, 16),
            admission_command=env.get("ARC_LANG_ADMISSION_COMMAND") or None,
        ),
        tts=TtsSettings(
            command=_command(env, "ARC_LANG_TTS_COMMAND"),
            remote=_flag(env, "ARC_LANG_TTS_REMOTE", False),
            timeout_seconds=float(tts_timeout) if tts_timeout is not None else 3600.0,
            remote_host=env.get("ARC_LANG_REMOTE_HOST") or None,
            remote_command=env.get("ARC_LANG_REMOTE_CLI") or None,
            audio_format="opus" if audio_format == "opus" else "wav",
            worker_token=_secret(env, "ARC_LANG_AUDIO_WORKER_TOKEN"),
            remote_url=(env.get("ARC_LANG_REMOTE_URL") or "").rstrip("/") or None,
            remote_token=_secret(env, "ARC_LANG_REMOTE_TOKEN"),
        ),
        auth=AuthSettings(
            mode=cast(AuthMode, auth_mode),
            password_hash=password_hash,
            proxy_header=env.get("ARC_LANG_AUTH_PROXY_HEADER", "X-Forwarded-User"),
            proxy_addresses=proxy_addresses,
            allowed_hosts=_list(env, "ARC_LANG_ALLOWED_HOSTS"),
        ),
    )

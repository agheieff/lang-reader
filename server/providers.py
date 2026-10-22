"""Model providers: the configuration file that names them and assigns them to stages.

``lang-reader.toml`` (or the file named by ``ARC_LANG_CONFIG``) lists named providers and says
which provider serves each generation stage. This module only parses and validates; making the
calls is ``model_callback.py``. API keys are never part of the file: a provider points at a key
file (``api_key_file``) or an environment variable (``api_key_env``), and the key is read once
here so a missing or empty key stops the worker at start.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[import-not-found]

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from server.settings import CALLBACK_STAGES

_NAME = re.compile(r"[A-Za-z0-9_-]+")
_URL = re.compile(r"https?://[^\s/]+(/\S*)?")
_TOP_LEVEL = {"providers", "stages"}


class _Provider(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = ""  # filled from the table name by load_config
    timeout_seconds: int = Field(600, ge=1, le=3600)  # wait for one answer


class _HttpProvider(_Provider):
    base_url: str
    model: str = Field(min_length=1)
    api_key_file: str | None = None
    api_key_env: str | None = None
    max_output_tokens: int | None = Field(None, ge=256, le=1_000_000)
    extra_body: dict[str, Any] = Field(default_factory=dict)  # merged into the request JSON
    api_key: str | None = Field(None, exclude=True, repr=False)  # resolved, never from the file

    @model_validator(mode="before")
    @classmethod
    def _no_inline_key(cls, data: Any) -> Any:
        if isinstance(data, dict) and "api_key" in data:
            raise ValueError("an inline api_key is not accepted: use api_key_file or api_key_env")
        return data

    @field_validator("base_url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = value.rstrip("/")
        if not _URL.fullmatch(value):
            raise ValueError("must be an http(s) URL such as https://host/v1")
        return value


class OpenAIProvider(_HttpProvider):
    """Any server that answers ``POST {base_url}/chat/completions`` in the OpenAI format."""

    kind: Literal["openai"]
    schema_mode: Literal["schema", "json_object", "prompt"] = "schema"
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"


class AnthropicProvider(_HttpProvider):
    """The Anthropic Messages API (``POST {base_url}/v1/messages``)."""

    kind: Literal["anthropic"]
    base_url: str = "https://api.anthropic.com"
    max_output_tokens: int | None = Field(16000, ge=256, le=1_000_000)
    anthropic_version: str = "2023-06-01"


class CommandProvider(_Provider):
    """An external command that follows the protocol of ``agent_command.py``."""

    kind: Literal["command"]
    command: list[str] = Field(min_length=1)
    pass_env: list[str] = Field(default_factory=list)  # extra environment variables to pass on


Provider = OpenAIProvider | AnthropicProvider | CommandProvider
_PROVIDER: TypeAdapter[Provider] = TypeAdapter(Annotated[Provider, Field(discriminator="kind")])


@dataclass(frozen=True)
class ModelConfig:
    providers: Mapping[str, Provider]
    stages: Mapping[str, tuple[str, ...]]  # stage (or "default") -> provider names, best first

    def chain(self, stage: str) -> tuple[Provider, ...]:
        """The providers to try for a stage: the first, then its fallbacks in order."""

        names = self.stages.get(stage) or self.stages["default"]
        return tuple(self.providers[name] for name in names)


def load_config(path: Path, env: Mapping[str, str] | None = None) -> ModelConfig:
    """Parse and validate the provider file; raises ValueError naming the problem."""

    env = os.environ if env is None else env
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"no provider configuration at {path}: copy lang-reader.example.toml to "
            "lang-reader.toml and edit it (README, 'Connect a model')"
        ) from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path} is not valid TOML: {error}") from error
    if unknown := set(raw) - _TOP_LEVEL:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown)}")
    tables = raw.get("providers")
    if not isinstance(tables, dict) or not tables:
        raise ValueError(f"{path}: define at least one [providers.NAME] table")
    providers = {name: _provider(path, name, table) for name, table in tables.items()}
    stages = _stages(path, raw.get("stages", {}), providers)
    # Keys are read only for providers a stage uses, so an example file can list spare ones.
    for name in {name for names in stages.values() for name in names}:
        provider = providers[name]
        if isinstance(provider, _HttpProvider):
            providers[name] = provider.model_copy(update={"api_key": _api_key(path, provider, env)})
    return ModelConfig(providers=providers, stages=stages)


def _provider(path: Path, name: str, table: Any) -> Provider:
    if not _NAME.fullmatch(name):
        raise ValueError(f"{path}: provider name {name!r} may use letters, digits, - and _ only")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: providers.{name} must be a table")
    try:
        provider = _PROVIDER.validate_python({**table, "name": name})
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in item['loc'][1:]) or 'kind'}: {item['msg']}"
            for item in error.errors()
        )
        raise ValueError(f"{path}: providers.{name}: {details}") from error
    return provider


def _api_key(path: Path, provider: _HttpProvider, env: Mapping[str, str]) -> str | None:
    if provider.api_key_file and provider.api_key_env:
        raise ValueError(f"{path}: providers.{provider.name}: use api_key_file or api_key_env")
    if provider.api_key_file:
        key_path = Path(provider.api_key_file).expanduser()
        if not key_path.is_absolute():
            key_path = path.parent / key_path  # relative to the configuration file
        try:
            key = key_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError(
                f"{path}: providers.{provider.name}: cannot read api_key_file {key_path}: "
                f"{error.strerror}"
            ) from error
        source = f"api_key_file {key_path}"
    elif provider.api_key_env:
        key = env.get(provider.api_key_env, "").strip()
        source = f"api_key_env {provider.api_key_env}"
    elif isinstance(provider, AnthropicProvider):
        raise ValueError(f"{path}: providers.{provider.name}: anthropic needs a key")
    else:
        return None  # a local server may need none
    if not key:
        raise ValueError(f"{path}: providers.{provider.name}: {source} is empty")
    return key


def _stages(
    path: Path, table: Any, providers: Mapping[str, Provider]
) -> dict[str, tuple[str, ...]]:
    if not isinstance(table, dict):
        raise ValueError(f"{path}: [stages] must be a table")
    stages: dict[str, tuple[str, ...]] = {}
    for stage, value in table.items():
        if stage != "default" and stage not in CALLBACK_STAGES:
            raise ValueError(
                f"{path}: stages.{stage}: unknown stage (use default or one of "
                f"{', '.join(CALLBACK_STAGES)})"
            )
        names = (
            (value,) if isinstance(value, str) else tuple(value) if isinstance(value, list) else ()
        )
        if not names or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{path}: stages.{stage} must be a provider name or a list of them")
        if missing := [name for name in names if name not in providers]:
            raise ValueError(f"{path}: stages.{stage} names unknown providers {missing}")
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: stages.{stage} lists a provider twice")
        stages[stage] = names
    if "default" not in stages:
        if len(providers) != 1:
            raise ValueError(f"{path}: set stages.default (several providers are defined)")
        stages["default"] = (next(iter(providers)),)
    return stages

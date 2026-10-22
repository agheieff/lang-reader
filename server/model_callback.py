"""The generation callback: route each stage to its configured provider and run it.

A stage may name several providers; they are tried in order until one succeeds, so a cheap
provider can back up a strong one. Only a failed or unavailable call falls through: an answer
that arrives but does not validate is a failed draft, handled by the worker's own retry.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

from server import settings as settings_module
from server.agent_command import run_command
from server.generation_callback import CallbackInvocation, CallbackResult
from server.model_http import Unavailable, ask_anthropic, ask_openai
from server.providers import (
    AnthropicProvider,
    CommandProvider,
    ModelConfig,
    OpenAIProvider,
    Provider,
    load_config,
)


def ask(
    provider: OpenAIProvider | AnthropicProvider, prompt: str, schema: dict[str, Any]
) -> tuple[str, dict[str, int]]:
    if isinstance(provider, OpenAIProvider):
        return ask_openai(provider, prompt, schema)
    return ask_anthropic(provider, prompt, schema)


class ProviderCallback:
    def __init__(self, config: ModelConfig, unavailable_exit_code: int) -> None:
        self.config = config
        self.unavailable_exit_code = unavailable_exit_code

    def run(self, invocation: CallbackInvocation) -> CallbackResult:
        logs: list[str] = []
        result: CallbackResult | None = None
        for provider in self.config.chain(invocation.request.stage):
            result = self._run_provider(provider, invocation)
            logs.append(f"[provider {provider.name}] exit {result.exit_code}\n{result.log}")
            if result.exit_code == 0:
                break
        assert result is not None
        return replace(
            result,
            log="\n\n".join(logs),
            deferred=result.exit_code == self.unavailable_exit_code,
        )

    def _run_provider(self, provider: Provider, invocation: CallbackInvocation) -> CallbackResult:
        if isinstance(provider, CommandProvider):
            return run_command(
                list(provider.command),
                invocation=invocation,
                timeout_seconds=provider.timeout_seconds,
                passthrough=tuple(provider.pass_env),
            )
        assert invocation.prompt_path is not None
        prompt = invocation.prompt_path.read_text(encoding="utf-8")
        schema = json.loads(invocation.response_schema_path.read_text(encoding="utf-8"))
        try:
            text, usage = ask(provider, prompt, schema)
        except Unavailable as error:
            return CallbackResult(exit_code=self.unavailable_exit_code, payload="", log=str(error))
        except RuntimeError as error:
            return CallbackResult(exit_code=1, payload="", log=str(error))
        return CallbackResult(exit_code=0, payload=text, log="ok", usage=usage)


def load_callback() -> ProviderCallback:
    """Build the callback from the provider file; raises ValueError when it is unusable."""

    agent = settings_module.load().agent
    return ProviderCallback(load_config(agent.config_path), agent.unavailable_exit_code)


_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def probe_provider(provider: Provider) -> dict[str, Any]:
    """Send one tiny request (`lang provider test`); raises ValueError with the reason."""

    if isinstance(provider, CommandProvider):
        raise ValueError("a command provider cannot be probed; run a real generation instead")
    started = time.perf_counter()
    try:
        text, usage = ask(provider, 'Reply with the JSON object {"ok": true}.', _PROBE_SCHEMA)
    except (Unavailable, RuntimeError) as error:
        raise ValueError(f"{provider.name}: {error}") from error
    try:
        answered = json.loads(text)
    except ValueError:
        answered = None
    return {
        "provider": provider.name,
        "ok": isinstance(answered, dict) and answered.get("ok") is True,
        "seconds": round(time.perf_counter() - started, 2),
        "answer": text[:200],
        "usage": usage,
    }

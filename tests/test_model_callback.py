from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from server.generation_callback import CallbackInvocation
from server.model_callback import ProviderCallback, probe_provider
from server.providers import AnthropicProvider, CommandProvider, ModelConfig, OpenAIProvider

KEY = "sk-test-secret-key"


class Endpoint:
    """A stand-in model server that records what it was asked and answers as told."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.reply: dict[str, Any] = {}
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                endpoint.requests.append({"path": self.path, "headers": self.headers, "body": body})
                self.send_response(endpoint.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(endpoint.reply).encode())

            def log_message(self, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def endpoint() -> Iterator[Endpoint]:
    server = Endpoint()
    yield server
    server.server.shutdown()
    server.server.server_close()


OPENAI_REPLY = {
    "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "prompt_tokens_details": {"cached_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 5},
    },
}
ANTHROPIC_REPLY = {
    "content": [{"type": "text", "text": '{"ok":'}, {"type": "text", "text": "true}"}],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 20,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 7,
        "output_tokens": 30,
    },
}


def _openai(url: str, key: bool = True, **extra: Any) -> OpenAIProvider:
    fields: dict[str, Any] = {
        "kind": "openai",
        "name": "oa",
        "base_url": f"{url}/v1",
        "model": "test-model",
        **extra,
    }
    return OpenAIProvider(**fields).model_copy(update={"api_key": KEY if key else None})


def _anthropic(url: str, **extra: Any) -> AnthropicProvider:
    fields: dict[str, Any] = {"name": "an", **extra}
    provider = AnthropicProvider(kind="anthropic", base_url=url, model="claude-test", **fields)
    return provider.model_copy(update={"api_key": KEY})


def _callback(providers: dict[str, Any], stages: dict[str, tuple[str, ...]]) -> ProviderCallback:
    return ProviderCallback(ModelConfig(providers=providers, stages=stages), 3)


def _only(provider: Any) -> ProviderCallback:
    return _callback({provider.name: provider}, {"default": (provider.name,)})


def test_openai_request_answer_and_usage(
    invocation: CallbackInvocation,
    endpoint: Endpoint,
) -> None:
    endpoint.reply = OPENAI_REPLY
    result = _only(_openai(endpoint.url, max_output_tokens=4000)).run(invocation)
    assert (result.exit_code, result.payload, result.deferred) == (0, '{"ok":true}', False)
    assert result.usage == {
        "input_tokens": 120,
        "cached_input_tokens": 100,
        "output_tokens": 30,
        "reasoning_output_tokens": 5,
    }
    request = endpoint.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"
    body = request["body"]
    assert body["model"] == "test-model" and body["max_tokens"] == 4000
    assert body["messages"][0]["content"].startswith("Write the frozen lesson prose")
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["type"] == "object"


@pytest.mark.parametrize("mode", ["prompt", "json_object"])
def test_openai_prompt_modes_append_the_schema(
    invocation: CallbackInvocation,
    endpoint: Endpoint,
    mode: str,
) -> None:
    endpoint.reply = OPENAI_REPLY
    provider = _openai(
        endpoint.url,
        schema_mode=mode,
        max_output_tokens=900,
        max_tokens_param="max_completion_tokens",
        extra_body={"temperature": 0.2},
        key=False,
    )
    assert _only(provider).run(invocation).exit_code == 0
    request = endpoint.requests[0]
    body = request["body"]
    assert "Authorization" not in request["headers"]  # a local server may take no key
    assert "\n\nOutput JSON schema:\n{" in body["messages"][0]["content"]
    assert body.get("response_format") == (
        {"type": "json_object"} if mode == "json_object" else None
    )
    assert body["max_completion_tokens"] == 900 and "max_tokens" not in body
    assert body["temperature"] == 0.2


def test_anthropic_request_answer_and_usage(
    invocation: CallbackInvocation,
    endpoint: Endpoint,
) -> None:
    endpoint.reply = ANTHROPIC_REPLY
    result = _only(_anthropic(endpoint.url)).run(invocation)
    assert (result.exit_code, result.payload) == (0, '{"ok":true}')
    # Anthropic keeps cached input apart from input_tokens; the app reports one input total.
    assert result.usage == {
        "input_tokens": 127,
        "cached_input_tokens": 100,
        "cache_creation_input_tokens": 7,
        "output_tokens": 30,
    }
    request = endpoint.requests[0]
    assert request["path"] == "/v1/messages"
    assert request["headers"]["x-api-key"] == KEY
    assert request["headers"]["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in request["headers"]
    body = request["body"]
    assert body["model"] == "claude-test" and body["max_tokens"] == 16000
    assert "Output JSON schema:" in body["messages"][0]["content"]
    assert "response_format" not in body


@pytest.mark.parametrize("make", [_openai, _anthropic])
@pytest.mark.parametrize("status", [429, 503])
def test_overload_is_unavailable_not_a_failure(
    invocation: CallbackInvocation,
    endpoint: Endpoint,
    make: Any,
    status: int,
) -> None:
    endpoint.status = status
    result = _only(make(endpoint.url)).run(invocation)
    assert result.exit_code == 3 and result.deferred and result.usage == {}
    assert str(status) in result.log


@pytest.mark.parametrize("make", [_openai, _anthropic])
def test_an_unreachable_server_is_unavailable(
    invocation: CallbackInvocation,
    make: Any,
) -> None:
    result = _only(make("http://127.0.0.1:9")).run(invocation)
    assert result.exit_code == 3 and result.deferred
    assert "unreachable" in result.log


@pytest.mark.parametrize("make", [_openai, _anthropic])
def test_other_http_errors_fail_and_never_echo_the_key(
    invocation: CallbackInvocation, endpoint: Endpoint, make: Any
) -> None:
    endpoint.status = 400
    endpoint.reply = {"error": {"message": f"Bad request near key {KEY}"}}
    result = _only(make(endpoint.url)).run(invocation)
    assert result.exit_code == 1 and not result.deferred
    assert "HTTP 400" in result.log and KEY not in result.log + result.payload
    assert "***" in result.log


@pytest.mark.parametrize("make", [_openai, _anthropic])
@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_is_reported_without_the_servers_text(
    invocation: CallbackInvocation, endpoint: Endpoint, make: Any, status: int
) -> None:
    endpoint.status = status
    endpoint.reply = {"error": {"message": f"Incorrect API key provided: {KEY[:6]}***{KEY[-3:]}"}}
    result = _only(make(endpoint.url)).run(invocation)
    assert result.exit_code == 1 and not result.deferred
    assert f"HTTP {status}" in result.log and KEY[:6] not in result.log


@pytest.mark.parametrize(
    "make,reply,message",
    [
        (_openai, {"choices": []}, "choices[0].message.content"),
        (_openai, {"choices": [{"message": {"content": "  "}}]}, "empty content"),
        (
            _openai,
            {"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]},
            "cut off",
        ),
        (_anthropic, {"content": "no"}, "no content blocks"),
        (_anthropic, {"content": [{"type": "thinking", "thinking": "x"}]}, "empty text"),
        (
            _anthropic,
            {"content": [{"type": "text", "text": "{"}], "stop_reason": "max_tokens"},
            "cut off",
        ),
    ],
)
def test_an_unusable_answer_is_a_failure(
    invocation: CallbackInvocation,
    endpoint: Endpoint,
    make: Any,
    reply: dict[str, Any],
    message: str,
) -> None:
    endpoint.reply = reply
    result = _only(make(endpoint.url)).run(invocation)
    assert result.exit_code == 1 and message in result.log


def _script(tmp_path: Path, body: str) -> CommandProvider:
    path = tmp_path / "agent.py"
    path.write_text(body, encoding="utf-8")
    return CommandProvider(kind="command", name="cmd", command=[sys.executable, str(path)])


def test_each_stage_uses_its_own_provider(
    invocation: CallbackInvocation,
    endpoint: Endpoint,
    tmp_path: Path,
) -> None:
    endpoint.reply = OPENAI_REPLY
    providers = {
        "http": _openai(endpoint.url, name="http"),
        "cmd": _script(tmp_path, "print('from-command')"),
    }
    prose_on_command = _callback(providers, {"default": ("http",), "prose": ("cmd",)})
    assert prose_on_command.run(invocation).payload.strip() == "from-command"
    assert endpoint.requests == []
    elsewhere = _callback(providers, {"default": ("http",), "lexical": ("cmd",)})
    assert elsewhere.run(invocation).payload == '{"ok":true}'  # the fixture's stage is prose
    assert len(endpoint.requests) == 1


def test_a_fallback_serves_when_the_first_is_unavailable_or_failing(
    invocation: CallbackInvocation,
    endpoint: Endpoint,
    tmp_path: Path,
) -> None:
    endpoint.reply = OPENAI_REPLY
    providers = {
        "down": _anthropic("http://127.0.0.1:9", name="down"),
        "broken": _script(tmp_path, "import sys; sys.exit(1)"),
        "up": _openai(endpoint.url, name="up"),
    }
    result = _callback(providers, {"default": ("down", "broken", "up")}).run(invocation)
    assert result.exit_code == 0 and result.payload == '{"ok":true}' and not result.deferred
    assert result.log.index("[provider down]") < result.log.index("[provider up]")
    assert "unreachable" in result.log


def test_a_fallback_is_not_tried_after_success(
    invocation: CallbackInvocation,
    endpoint: Endpoint,
    tmp_path: Path,
) -> None:
    endpoint.reply = OPENAI_REPLY
    providers = {"first": _openai(endpoint.url, name="first"), "second": _script(tmp_path, "1/0")}
    result = _callback(providers, {"default": ("first", "second")}).run(invocation)
    assert result.exit_code == 0 and "second" not in result.log


def test_every_provider_unavailable_defers_the_task(
    invocation: CallbackInvocation,
    endpoint: Endpoint,
) -> None:
    endpoint.status = 429
    providers = {
        "a": _openai(endpoint.url, name="a"),
        "b": _anthropic("http://127.0.0.1:9", name="b"),
    }
    result = _callback(providers, {"default": ("a", "b")}).run(invocation)
    assert result.exit_code == 3 and result.deferred


def test_a_slow_server_times_out_as_a_failure(
    invocation: CallbackInvocation,
    tmp_path: Path,
) -> None:
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)  # accepts the connection (backlog) but never answers
    try:
        provider = _openai(f"http://127.0.0.1:{listener.getsockname()[1]}", timeout_seconds=1)
        result = _only(provider).run(invocation)
    finally:
        listener.close()
    assert result.exit_code == 1 and "timed out" in result.log


def test_probe_reports_the_answer_and_cannot_probe_a_command(
    endpoint: Endpoint, tmp_path: Path
) -> None:
    endpoint.reply = OPENAI_REPLY
    report = probe_provider(_openai(endpoint.url))
    assert report["ok"] is True and report["usage"]["input_tokens"] == 120
    assert KEY not in json.dumps(report)
    endpoint.status = 503
    with pytest.raises(ValueError, match="503"):
        probe_provider(_openai(endpoint.url))
    with pytest.raises(ValueError, match="command provider"):
        probe_provider(_script(tmp_path, "pass"))


def test_the_shipped_command_adapter_follows_the_protocol(
    invocation: CallbackInvocation, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint.reply = OPENAI_REPLY
    adapter = Path(__file__).resolve().parents[1] / "examples" / "command_adapter.py"
    provider = CommandProvider(
        kind="command",
        name="adapter",
        command=[sys.executable, str(adapter)],
        pass_env=["ADAPTER_URL", "ADAPTER_MODEL"],
    )
    monkeypatch.setenv("ADAPTER_URL", f"{endpoint.url}/v1/chat/completions")
    monkeypatch.setenv("ADAPTER_MODEL", "m")
    result = _only(provider).run(invocation)
    endpoint.status = 429
    busy = _only(provider).run(invocation)
    assert result.exit_code == 0 and result.payload == '{"ok":true}'
    assert result.usage == {"input_tokens": 120, "output_tokens": 30}
    assert busy.exit_code == 3 and busy.deferred

"""Calls to hosted and local model servers: OpenAI-compatible and Anthropic Messages.

Both kinds share one HTTP path (``post_json``); a kind only builds its request and reads its
answer. Failure semantics match the command protocol: ``Unavailable`` (unreachable, HTTP 429 or
503) means no capacity right now and nothing happened, so the task waits; any other failure is a
``RuntimeError`` and counts as a failed attempt. The API key is scrubbed from every message so it
cannot reach a log, whatever a server echoes back.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from server.providers import AnthropicProvider, OpenAIProvider

_UNAVAILABLE_STATUSES = {429, 503}
_AUTH_STATUSES = {401, 403}


class Unavailable(Exception):
    """No capacity right now and no effect happened."""


def post_json(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float, secret: str | None
) -> dict[str, Any]:
    def scrub(text: str) -> str:
        return text.replace(secret, "***") if secret else text

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code in _UNAVAILABLE_STATUSES:
            raise Unavailable(f"server answered HTTP {error.code}") from error
        if error.code in _AUTH_STATUSES:
            # Servers may echo part of the key back; say what is wrong without their text.
            raise RuntimeError(f"server refused the credentials (HTTP {error.code})") from error
        detail = scrub(error.read(500).decode("utf-8", "replace"))
        raise RuntimeError(f"server answered HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise RuntimeError("server timed out") from error
        raise Unavailable(scrub(f"server unreachable: {error.reason}")) from error
    except TimeoutError as error:
        raise RuntimeError("server timed out") from error
    except ConnectionError as error:
        raise Unavailable(scrub(f"connection lost: {error}")) from error
    except ValueError as error:
        raise RuntimeError("server answered something that is not JSON") from error
    if not isinstance(answer, dict):
        raise RuntimeError("server answered something other than a JSON object")
    return answer


def _with_schema(prompt: str, schema: dict[str, Any]) -> str:
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return f"{prompt}\n\nOutput JSON schema:\n{schema_text}"


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _usage(**fields: Any) -> dict[str, int]:
    return {name: count for name, value in fields.items() if (count := _count(value)) is not None}


def ask_openai(
    provider: OpenAIProvider, prompt: str, schema: dict[str, Any]
) -> tuple[str, dict[str, int]]:
    """One chat completion; returns the answer text and the usage in the app's field names."""

    body: dict[str, Any] = {"model": provider.model}
    if provider.schema_mode == "schema":
        content = prompt
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "lang_reader_response", "schema": schema},
        }
    else:
        content = _with_schema(prompt, schema)
        if provider.schema_mode == "json_object":
            body["response_format"] = {"type": "json_object"}
    body["messages"] = [{"role": "user", "content": content}]
    if provider.max_output_tokens is not None:
        body[provider.max_tokens_param] = provider.max_output_tokens
    body.update(provider.extra_body)
    headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}
    answer = post_json(
        f"{provider.base_url}/chat/completions",
        headers,
        body,
        provider.timeout_seconds,
        provider.api_key,
    )
    try:
        choice = answer["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("answer has no choices[0].message.content") from error
    if choice.get("finish_reason") == "length":
        raise RuntimeError("answer was cut off at the output token limit; raise max_output_tokens")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("answer has empty content")
    usage = answer.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    prompt_details = usage.get("prompt_tokens_details")
    completion_details = usage.get("completion_tokens_details")
    return text, _usage(
        input_tokens=usage.get("prompt_tokens"),
        cached_input_tokens=(
            prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None
        ),
        output_tokens=usage.get("completion_tokens"),
        reasoning_output_tokens=(
            completion_details.get("reasoning_tokens")
            if isinstance(completion_details, dict)
            else None
        ),
    )


def ask_anthropic(
    provider: AnthropicProvider, prompt: str, schema: dict[str, Any]
) -> tuple[str, dict[str, int]]:
    """One Messages API call; the schema travels in the prompt."""

    body: dict[str, Any] = {
        "model": provider.model,
        "max_tokens": provider.max_output_tokens,
        "messages": [{"role": "user", "content": _with_schema(prompt, schema)}],
        **provider.extra_body,
    }
    answer = post_json(
        f"{provider.base_url}/v1/messages",
        {"x-api-key": str(provider.api_key), "anthropic-version": provider.anthropic_version},
        body,
        provider.timeout_seconds,
        provider.api_key,
    )
    blocks = answer.get("content")
    if not isinstance(blocks, list):
        raise RuntimeError("answer has no content blocks")
    text = "".join(
        block["text"]
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if answer.get("stop_reason") == "max_tokens":
        raise RuntimeError("answer was cut off at the output token limit; raise max_output_tokens")
    if not text.strip():
        raise RuntimeError("answer has empty text")
    usage = answer.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    plain = _count(usage.get("input_tokens"))
    read = _count(usage.get("cache_read_input_tokens"))
    written = _count(usage.get("cache_creation_input_tokens"))
    # Anthropic reports cached input apart from input_tokens; the app counts all input in one
    # number (as OpenAI does) with the cached part also listed.
    total = None if plain is None else plain + (read or 0) + (written or 0)
    return text, _usage(
        input_tokens=total,
        cached_input_tokens=read,
        cache_creation_input_tokens=written,
        output_tokens=usage.get("output_tokens"),
    )

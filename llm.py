"""Minimal LLM wrapper around OpenAI-compatible chat-completions endpoints.

Config via environment variables (see .env.example):
  OPENAI_API_KEY     - API key
  OPENAI_BASE_URL    - endpoint base, defaults to https://api.openai.com/v1
  OPENAI_MODEL       - model id, defaults to gpt-4o-mini

The interface is intentionally tiny: chat() for plain text, chat_json() for
structured policy output. Swapping in another provider only requires
reimplementing this module.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    """True for rate-limit / server errors that are safe to retry."""
    return getattr(exc, "status_code", None) in _TRANSIENT_STATUS


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        if not self.api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Provide --api-key or set the env var."
            )
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        retries: int = 3,
    ) -> tuple[str, dict[str, Any]]:
        """Single non-streaming completion. Returns (text, meta).

        Transient provider errors (429 / 5xx) are retried with exponential
        backoff; other errors surface immediately.
        """
        start = time.monotonic()
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                tokens = usage.total_tokens if usage else None
                return text, {"ms": round((time.monotonic() - start) * 1000), "tokens": tokens}
            except Exception as exc:  # noqa: BLE001 - surface provider errors as LLMError
                last_exc = exc
                if not _is_transient(exc) or attempt >= retries:
                    break
                time.sleep(min(2**attempt, 8) + random.uniform(0, 0.5))
        raise LLMError(f"chat completion failed: {last_exc}") from last_exc

    def chat_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        retries: int = 1,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Completion with a JSON-object constraint. Returns (parsed, meta).

        On invalid/truncated JSON, retries once with a corrective prompt
        that re-feeds the failed output back to the model.
        """
        constrained = [*messages, {"role": "user", "content": "Return only valid JSON."}]
        text, meta = self.chat(constrained, max_tokens=max_tokens)
        for attempt in range(retries + 1):
            cleaned = text.strip()
            # strip markdown fences if the model wraps the JSON
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                if attempt >= retries:
                    raise LLMError(f"model returned non-JSON: {cleaned[:200]!r}") from exc
                constrained = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Your previous output was truncated or not valid JSON. "
                                                  "Return ONLY valid JSON, complete."},
                ]
                text, meta = self.chat(constrained, max_tokens=max_tokens)
                continue
            if not isinstance(parsed, dict):
                if attempt >= retries:
                    raise LLMError(f"model returned non-object JSON: {parsed!r}")
                continue
            return parsed, meta
        raise LLMError("model returned non-object JSON")

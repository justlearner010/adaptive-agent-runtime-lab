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
import time
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


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
    ) -> str:
        """Single non-streaming completion. Returns assistant text."""
        start = time.monotonic()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # surface provider errors as LLMError
            raise LLMError(f"chat completion failed: {exc}") from exc
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        tokens = usage.total_tokens if usage else None
        return text, {"ms": round((time.monotonic() - start) * 1000), "tokens": tokens}

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

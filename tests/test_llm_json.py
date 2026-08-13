"""Tests for chat_json retry-on-truncation behavior."""

import pytest

from llm import LLM, LLMError


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Usage:
    total_tokens = 10


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _Completions:
    def __init__(self, queue: list[str]) -> None:
        self._queue = queue
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _Resp(self._queue.pop(0))


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, queue: list[str]) -> None:
        self.chat = _Chat(_Completions(queue))


def _llm_with(queue: list[str]) -> tuple[LLM, _Completions]:
    llm = LLM(api_key="test-key")  # no network call at construction
    llm._client = _FakeClient(queue)  # type: ignore[attr-defined]
    return llm, llm._client.chat.completions  # type: ignore[attr-defined]


def test_retry_recovers_from_truncated_json():
    truncated = '{"subtasks":[{"title":"Explain ReAct"'
    valid = '{"subtasks":[{"title":"Explain ReAct","prompt":"Describe ReAct."}]}'
    llm, completions = _llm_with([truncated, valid])

    parsed, _ = llm.chat_json([{"role": "user", "content": "task"}])

    assert parsed == {"subtasks": [{"title": "Explain ReAct", "prompt": "Describe ReAct."}]}
    assert completions.calls == 2


def test_persistent_invalid_json_raises_after_retries():
    llm, completions = _llm_with(["not json at all", "still not json"])

    with pytest.raises(LLMError):
        llm.chat_json([{"role": "user", "content": "task"}])
    assert completions.calls == 2


def test_valid_json_first_try_no_retry():
    llm, completions = _llm_with(['{"ok": true}'])

    parsed, _ = llm.chat_json([{"role": "user", "content": "task"}])

    assert parsed == {"ok": True}
    assert completions.calls == 1

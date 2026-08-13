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


# --- transient error retry (429/5xx) ---

class _StatusError(Exception):
    def __init__(self, status: int) -> None:
        self.status_code = status
        super().__init__(f"status {status}")


class _RaisingCompletions:
    def __init__(self, fail_status: int, fail_times: int, content: str = "ok") -> None:
        self._fail_status = fail_status
        self._fail_left = fail_times
        self._content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self._fail_left > 0:
            self._fail_left -= 1
            raise _StatusError(self._fail_status)
        return _Resp(self._content)


class _RaisingClient:
    def __init__(self, completions: _RaisingCompletions) -> None:
        self.chat = _Chat(completions)


def test_transient_429_is_retried():
    llm = LLM(api_key="test-key")
    comp = _RaisingCompletions(fail_status=429, fail_times=2)
    llm._client = _RaisingClient(comp)  # type: ignore[attr-defined]

    text, _ = llm.chat([{"role": "user", "content": "hi"}], retries=3)

    assert text == "ok"
    assert comp.calls == 3


def test_non_transient_error_not_retried():
    llm = LLM(api_key="test-key")
    comp = _RaisingCompletions(fail_status=401, fail_times=99)
    llm._client = _RaisingClient(comp)  # type: ignore[attr-defined]

    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "hi"}], retries=3)
    assert comp.calls == 1


def test_retries_exhausted_raises():
    llm = LLM(api_key="test-key")
    comp = _RaisingCompletions(fail_status=503, fail_times=99)
    llm._client = _RaisingClient(comp)  # type: ignore[attr-defined]

    with pytest.raises(LLMError):
        llm.chat([{"role": "user", "content": "hi"}], retries=2)
    assert comp.calls == 3

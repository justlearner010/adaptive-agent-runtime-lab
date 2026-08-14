"""Tests for the ReAct executor loop (empty-response retry robustness)."""

from executors.react import ReactExecutor, _parse_action
from trace import Trace


class FakeLLM:
    """Scripted LLM: pops (text, meta) responses in order."""

    def __init__(self, responses: list[tuple[str, dict]]):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, max_tokens=None, retries=3, response_format=None):
        self.calls += 1
        return self._responses.pop(0)


def _run(responses: list[tuple[str, dict]], max_steps: int = 3):
    llm = FakeLLM(responses)
    trace = Trace()
    answer = ReactExecutor(max_steps=max_steps).execute("test task", llm, trace)
    return llm, trace, answer


def test_empty_first_response_is_retried():
    llm, _, answer = _run([("", {"ms": 1, "tokens": 0}), ("Final Answer: 42", {"ms": 2, "tokens": 9})])
    assert llm.calls == 2
    assert answer.text == "42"


def test_empty_middle_step_retried_then_continues():
    # action -> (empty response) -> final answer
    responses = [
        ('Action: {"tool": "calculator", "input": "2+2"}', {"ms": 1, "tokens": 5}),
        ("", {"ms": 1, "tokens": 0}),
        ("Final Answer: 4", {"ms": 1, "tokens": 3}),
    ]
    llm, _, answer = _run(responses)
    assert llm.calls == 3
    assert answer.text == "4"


def test_all_empty_responses_give_no_answer_and_warn():
    llm, trace, answer = _run([("", {"ms": 1, "tokens": 0})] * 3)
    assert llm.calls == 3  # 1 step x (initial + 2 retries)
    assert answer.text == "No answer produced."
    warnings = [e for e in trace.to_dict() if e["kind"] == "llm_call" and "warning" in e["data"]]
    assert len(warnings) == 3


def test_parse_action_valid():
    assert _parse_action('Action: {"tool": "search", "input": "x"}') == {"tool": "search", "input": "x"}


def test_parse_action_invalid():
    assert _parse_action("just some text") is None

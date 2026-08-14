"""Tests for ReAct Action extraction (brace-aware, no greedy regex)."""

from executors.react import ReactExecutor, _parse_action
from trace import Trace


def test_single_action():
    assert _parse_action('Thought: hi\nAction: {"tool": "search", "input": "x"}') == {
        "tool": "search",
        "input": "x",
    }


def test_action_with_trailing_text_and_braces():
    # The old greedy regex would swallow from '{' to the last '}' and fail to parse.
    text = 'Action: {"tool": "calculator", "input": "2+2"} | extra } text }'
    assert _parse_action(text) == {"tool": "calculator", "input": "2+2"}


def test_multiple_actions_uses_first():
    text = (
        'Action: {"tool": "search", "input": "a"} | Observation: nothing. '
        'Action: {"tool": "search", "input": "b"}'
    )
    assert _parse_action(text) == {"tool": "search", "input": "a"}


def test_truncated_json_returns_none():
    assert _parse_action('Action: {"tool": "search", "input": "x') is None


def test_no_action_returns_none():
    assert _parse_action("Final Answer: done") is None


def test_nested_braces_in_input():
    text = 'Action: {"tool": "search", "input": "{\\"k\\": 1}"}'
    assert _parse_action(text) == {"tool": "search", "input": '{"k": 1}'}


def test_unmatched_brace_inside_string_value():
    # braces inside JSON strings must not be counted as structural
    text = 'Action: {"tool": "search", "input": "find {"}'
    assert _parse_action(text) == {"tool": "search", "input": "find {"}


def test_closing_brace_inside_string_value():
    text = 'Action: {"tool": "search", "input": "a}b"}'
    assert _parse_action(text) == {"tool": "search", "input": "a}b"}


def test_executor_executes_tool_despite_trailing_garbage():
    class FakeLLM:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = 0

        def chat(self, messages, max_tokens=None, retries=3, response_format=None):
            self.calls += 1
            return self._responses.pop(0)

    llm = FakeLLM(
        [
            ('Action: {"tool": "calculator", "input": "2+2"} | and then } more }', {"ms": 1, "tokens": 5}),
            ("Final Answer: 4", {"ms": 1, "tokens": 3}),
        ]
    )
    answer = ReactExecutor(max_steps=3).execute("calculate", llm, Trace())
    assert answer.tool_calls == [{"tool": "calculator", "args": {"input": "2+2"}}]
    assert answer.text == "4"

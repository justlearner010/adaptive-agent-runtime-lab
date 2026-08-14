"""Tests for subagent graceful degradation when the planner fails."""

from llm import LLMError
from executors.subagent import SubagentExecutor
from trace import Trace


class _PlannerFailsLLM:
    def chat_json(self, messages, **kwargs):
        raise LLMError("planner returned invalid JSON")

    def chat(self, messages, **kwargs):
        return "Final Answer: done", {"ms": 1, "tokens": 1}


class _NormalLLM:
    def chat_json(self, messages, **kwargs):
        return {"subtasks": [{"title": "a", "prompt": "do a"}, {"title": "b", "prompt": "do b"}]}, {"ms": 1, "tokens": 1}

    def chat(self, messages, **kwargs):
        return "Final Answer: done", {"ms": 1, "tokens": 1}


class _EmptySynthesisLLM:
    """Workers produce reports; the synthesizer returns an empty string."""

    def chat_json(self, messages, **kwargs):
        return {"subtasks": [{"title": "a", "prompt": "do a"}]}, {"ms": 1, "tokens": 1}

    def chat(self, messages, **kwargs):
        # first call = worker react step, second call = synthesizer (empty)
        if self._calls < 1:
            self._calls += 1
            return "Final Answer: worker report", {"ms": 1, "tokens": 1}
        self._calls += 1
        return "", {"ms": 1, "tokens": 1}

    def __init__(self) -> None:
        self._calls = 0

class _TitleOnlyEmptySynthesisLLM:
    """Planner returns a subtask with title only (title becomes the prompt);
    the synthesizer returns empty so the fallback path is exercised."""

    def __init__(self) -> None:
        self._calls = 0

    def chat_json(self, messages, **kwargs):
        return {"subtasks": [{"title": "do internal step"}]}, {"ms": 1, "tokens": 1}

    def chat(self, messages, **kwargs):
        if self._calls < 1:
            self._calls += 1
            return "Final Answer: done", {"ms": 1, "tokens": 1}
        self._calls += 1
        return "", {"ms": 1, "tokens": 1}


def test_planner_failure_degrades_to_single_subtask():
    executor = SubagentExecutor(max_steps_per_agent=1)
    trace = Trace()
    answer = executor.execute("some task", _PlannerFailsLLM(), trace)

    assert answer.text == "Final Answer: done"
    assert answer.strategy == "subagent"
    spawns = [e for e in trace.to_dict() if e["kind"] == "subagent"]
    assert len(spawns) == 1
    assert spawns[0]["data"]["subtask"] == "some task"


def test_normal_plan_uses_all_subtasks():
    executor = SubagentExecutor(max_steps_per_agent=1)
    trace = Trace()
    answer = executor.execute("some task", _NormalLLM(), trace)

    assert answer.text == "Final Answer: done"
    spawns = [e for e in trace.to_dict() if e["kind"] == "subagent"]
    assert len(spawns) == 2


def test_empty_synthesis_falls_back_to_worker_reports():
    executor = SubagentExecutor(max_steps_per_agent=1)
    trace = Trace()
    answer = executor.execute("some task", _EmptySynthesisLLM(), trace)

    assert "worker report" in answer.text
    # internal subtask prompt must not leak into the user-facing answer
    assert "do a" not in answer.text
    synth_events = [e for e in trace.to_dict() if e["kind"] == "llm_call" and "synthesizer" in (e["data"].get("role") or "")]
    assert synth_events[-1]["data"].get("error", "").startswith("empty synthesis")


def test_title_as_prompt_does_not_leak_in_fallback():
    executor = SubagentExecutor(max_steps_per_agent=1)
    trace = Trace()
    answer = executor.execute("some task", _TitleOnlyEmptySynthesisLLM(), trace)

    assert "done" in answer.text
    # the title doubled as the worker prompt and must not surface in the answer
    assert "do internal step" not in answer.text

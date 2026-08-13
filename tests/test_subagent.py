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

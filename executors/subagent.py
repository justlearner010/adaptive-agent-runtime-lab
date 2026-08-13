"""Subagent executor: decompose -> delegate -> synthesize.

v1 simplification: each subagent is a fresh ReAct loop over its subtask
(own message list, shared tool registry), so subtask contexts are isolated
from the parent. Parent then synthesizes the final answer from the
subagent reports with one more LLM call.
"""

from __future__ import annotations

import json

from llm import LLMError
from . import Answer
from .react import ReactExecutor

PLANNER_PROMPT = """You decompose the user's task into independent subtasks that can be
solved separately. Return ONLY JSON:

{"subtasks": [{"title": "<short title>", "prompt": "<self-contained instructions>"}, ...]}

Each subtask prompt must be self-contained: the worker agent will not see the original task.
Keep 2-4 subtasks unless the task is trivial.
"""

SYNTHESIZER_PROMPT = """You are the lead agent. The task was delegated to worker agents.
Here are their reports:

{reports}

Synthesize a single final answer to the task: {task}
"""


class SubagentExecutor:
    strategy = "subagent"

    def __init__(self, max_steps_per_agent: int = 6) -> None:
        self.max_steps_per_agent = max_steps_per_agent

    def execute(self, task: str, llm, trace) -> Answer:
        # 1) decompose
        try:
            plan, meta = llm.chat_json(
                [
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": task},
                ],
                max_tokens=800,
            )
            trace.record("llm_call", role="subagent/planner", **meta)
            subtasks = plan.get("subtasks", [])
        except LLMError:
            # planner refused/returned invalid JSON -> degrade to self-delegation
            trace.record("llm_call", role="subagent/planner", error="planner failed, falling back to single subtask")
            subtasks = []
        if not isinstance(subtasks, list) or not subtasks:
            subtasks = [{"title": "main", "prompt": task}]
        if not isinstance(subtasks, list) or not subtasks:
            subtasks = [{"title": "main", "prompt": task}]

        # 2) delegate: each subtask runs in an isolated ReAct context
        reports: list[str] = []
        for index, subtask in enumerate(subtasks):
            prompt = subtask.get("prompt") or subtask.get("title") or task
            title = subtask.get("title", f"subtask-{index}")
            trace.record("subagent", index=index, subtask=prompt)
            worker = ReactExecutor(max_steps=self.max_steps_per_agent)
            answer = worker.execute(prompt, llm, trace)
            reports.append(f"[{title}]\n{prompt}\n-> {answer.text}")

        # 3) synthesize
        text, meta = llm.chat(
            [
                {"role": "system", "content": SYNTHESIZER_PROMPT.format(reports="\n\n".join(reports), task=task)},
                {"role": "user", "content": "Produce the final answer."},
            ],
            max_tokens=600,
        )
        trace.record("llm_call", role="subagent/synthesizer", **meta)
        if not text.strip():
            # empty synthesis -> fall back to raw worker reports
            text = "\n\n".join(reports)
            trace.record("llm_call", role="subagent/synthesizer", error="empty synthesis, fell back to worker reports")

        return Answer(
            text=text,
            strategy=self.strategy,
            steps=len(subtasks) + 1,
            tool_calls=[],
        )

"""Subagent executor: decompose -> delegate (in parallel) -> synthesize.

Each subagent is a fresh ReAct loop over its subtask (own message list,
shared tool registry), so subtask contexts are isolated from the parent.
Since v2, workers run concurrently (ThreadPoolExecutor) so parallelizable
tasks (e.g. paging a long document) actually benefit; results are reordered
to subtask order for a deterministic final report.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from llm import LLMError
from . import Answer
from .react import ReactExecutor

PLANNER_PROMPT = """You decompose the user's task into independent subtasks that can be
solved separately. Return ONLY JSON:

{"subtasks": [{"title": "<short title>", "prompt": "<self-contained instructions>"}, ...]}

Each subtask prompt must be self-contained: the worker agent will not see the original task.
Keep 2-4 subtasks unless the task is trivial.

If the task involves reading a LONG document with the doc tool (each read returns at
most 2500 characters), split the document into char-offset ranges across subtasks so
the workers read different parts in parallel, e.g.:
- "Read document 'X' starting at offset 0 (read 2500 chars at a time until you reach
  '(end of X)' or offset 5000) and report every fact you find"
- "Read document 'X' starting at offset 5000 (read 2500 chars at a time until the end)
  and report every fact you find"
The synthesizer will merge the workers' findings, so each worker only needs its own range.
"""

SYNTHESIZER_PROMPT = """You are the lead agent. The task was delegated to worker agents.
Here are their reports:

{reports}

Synthesize a single final answer to the task: {task}
"""


class SubagentExecutor:
    strategy = "subagent"

    def __init__(self, max_steps_per_agent: int = 6, workers: int = 2) -> None:
        self.max_steps_per_agent = max_steps_per_agent
        self.workers = workers

    def execute(self, task: str, llm, trace) -> Answer:
        # 1) decompose
        try:
            plan, meta = llm.chat_json(
                [
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": task},
                ],
                max_tokens=2000,  # reasoning models need headroom before emitting JSON
            )
            trace.record("llm_call", role="subagent/planner", **meta)
            subtasks = plan.get("subtasks", [])
        except LLMError:
            # planner refused/returned invalid JSON -> degrade to self-delegation
            trace.record("llm_call", role="subagent/planner", error="planner failed, falling back to single subtask")
            subtasks = []
        if not isinstance(subtasks, list) or not subtasks:
            subtasks = [{"title": "main", "prompt": task}]

        # 2) delegate: each subtask runs in an isolated ReAct context.
        #    Spawn events are recorded up front (deterministic order), then
        #    workers run concurrently; results are reordered by subtask index.
        prepared: list[tuple[int, str, str, dict]] = []
        for index, subtask in enumerate(subtasks):
            prompt = subtask.get("prompt") or subtask.get("title") or task
            title = subtask.get("title", f"subtask-{index}")
            trace.record("subagent", index=index, subtask=prompt)
            prepared.append((index, title, prompt, subtask))

        def _run(item: tuple[int, str, str, dict]) -> tuple[int, str, str, dict, Answer]:
            index, title, prompt, subtask = item
            worker = ReactExecutor(max_steps=self.max_steps_per_agent)
            answer = worker.execute(prompt, llm, trace)
            return index, title, prompt, subtask, answer

        if self.workers > 1 and len(prepared) > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                done = list(pool.map(_run, prepared))
        else:
            done = [_run(item) for item in prepared]
        done.sort(key=lambda r: r[0])  # deterministic subtask order

        reports: list[str] = []
        fallback_parts: list[str] = []
        for _index, title, prompt, subtask, answer in done:
            reports.append(f"[{title}]\n{prompt}\n-> {answer.text}")
            # fallback keeps only the answer; the title is a label only when it
            # did NOT double as the worker prompt (no internal instruction leaks)
            label = f"[{title}]" if subtask.get("prompt") else ""
            fallback_parts.append(f"{label}\n{answer.text}".strip())

        # 3) synthesize
        text, meta = llm.chat(
            [
                {"role": "system", "content": SYNTHESIZER_PROMPT.format(reports="\n\n".join(reports), task=task)},
                {"role": "user", "content": "Produce the final answer."},
            ],
            max_tokens=1200,  # reasoning models need headroom; 600 truncated to a mid-reasoning fragment
        )
        trace.record("llm_call", role="subagent/synthesizer", **meta)
        if not text.strip():
            # empty synthesis -> fall back to worker answers (internal prompts stripped)
            text = "\n\n".join(fallback_parts)
            trace.record("llm_call", role="subagent/synthesizer", error="empty synthesis, fell back to worker reports")

        return Answer(
            text=text,
            strategy=self.strategy,
            steps=len(subtasks) + 1,
            tool_calls=[],
        )

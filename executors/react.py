"""ReAct executor: Thought -> Action(tool) -> Observation loop.

The model emits actions as JSON so we can parse them robustly:

    Thought: <reasoning>
    Action: {"tool": "<name>", "input": "<args as JSON string>"}

Observations are appended to the message list, keeping the single
ReAct context. Loop until the model stops emitting actions or
max_steps is reached.
"""

from __future__ import annotations

import json
import re
import time

import tools
from . import Answer

SYSTEM_PROMPT = f"""You are an agent that solves tasks by interleaving reasoning and tool use (ReAct).

Available tools:
{tools.describe()}

Format every step as:

Thought: <your reasoning>
Action: {{"tool": "<tool name>", "input": "<arguments as JSON>"}}

After each Action you will receive an Observation. Keep going until you have
enough information, then respond with:

Final Answer: <your answer>

Never invent tool results. Never answer before using tools if the task
requires information you do not have (e.g. calculations, corpus facts).
"""

_ACTION_RE = re.compile(r"Action:\s*(\{.*\})", re.DOTALL)


def _parse_action(text: str) -> dict | None:
    match = _ACTION_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("tool"), str):
        return payload
    return None


class ReactExecutor:
    strategy = "react"

    def __init__(self, max_steps: int = 8) -> None:
        self.max_steps = max_steps

    def execute(self, task: str, llm, trace) -> Answer:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        tool_calls: list[dict] = []
        step = 0
        final_answer = ""

        while step < self.max_steps:
            step += 1
            text, meta = llm.chat(messages, max_tokens=500)
            trace.record("llm_call", role=f"react/step{step}", **meta)
            trace.record("step", n=step, max_steps=self.max_steps, summary=text[:80].replace("\n", " "))

            action = _parse_action(text)
            if action is None:
                # no (valid) action -> treat as final answer
                final_answer = text.split("Final Answer:", 1)[-1].strip() or text.strip()
                break

            tool_name = action["tool"]
            raw_args = action.get("input", "")
            args = _parse_args(raw_args)
            start = time.monotonic()
            result = tools.execute(tool_name, args)
            ms = round((time.monotonic() - start) * 1000)
            trace.record("tool_call", tool=tool_name, args=args, ms=ms, ok=not result.startswith("Error"))
            tool_calls.append({"tool": tool_name, "args": args})

            messages.append({"role": "assistant", "content": text})
            messages.append(
                {"role": "user", "content": f"Observation: {result}\nContinue. Either emit another Action or a Final Answer."}
            )
        else:
            final_answer = "Reached max steps without a final answer."

        if not final_answer:
            final_answer = "No answer produced."

        return Answer(
            text=final_answer,
            strategy=self.strategy,
            steps=step,
            tool_calls=tool_calls,
        )


def _parse_args(raw: str) -> dict:
    """input comes as a JSON string; be lenient (object, or bare value)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"input": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"input": raw}

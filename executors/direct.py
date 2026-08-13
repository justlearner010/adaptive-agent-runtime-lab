"""Direct executor: single LLM call, no tools."""

from __future__ import annotations

from . import Answer

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's task directly and concisely "
    "with the knowledge you have. Do not use any tools."
)


class DirectExecutor:
    strategy = "direct"

    def execute(self, task: str, llm, trace) -> Answer:
        text, meta = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ]
        )
        trace.record("llm_call", role="direct", **meta)
        return Answer(text=text, strategy=self.strategy, steps=1)

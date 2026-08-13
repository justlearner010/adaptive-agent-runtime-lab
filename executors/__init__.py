"""Execution strategies.

Each executor implements:
    execute(task: str, llm: LLM, trace: Trace) -> Answer

Answer carries the final text plus metadata (steps, tool calls) so the
pipeline and trace layer can report on what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Answer:
    text: str
    strategy: str
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class Executor(Protocol):
    strategy: str

    def execute(self, task: str, llm: Any, trace: Any) -> Answer: ...

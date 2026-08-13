"""Structured execution tracing for the Agent Runtime Lab.

Records every decision and side effect in the pipeline:
Task -> Policy -> Execution Strategy -> steps -> final answer.
The collected data is the basis for later strategy evaluation and
learned policy work (roadmap v4/v5).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    kind: str
    data: dict[str, Any]
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "data": self.data, "ts": round(self.ts, 4)}


class Trace:
    """Collects ordered events and renders a human-readable report."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, kind: str, **data: Any) -> None:
        self.events.append(TraceEvent(kind=kind, data=data))

    def to_dict(self) -> list[dict[str, Any]]:
        return [e.as_dict() for e in self.events]

    def report(self) -> str:
        lines: list[str] = []
        for e in self.events:
            head = f"[{e.kind}]"
            if e.kind == "policy":
                lines.append(
                    f"{head} strategy={e.data.get('strategy')} "
                    f"complexity={e.data.get('complexity')} "
                    f"tools={e.data.get('tools_needed')}"
                )
            elif e.kind == "dispatch":
                lines.append(f"{head} -> executor={e.data.get('executor')}")
            elif e.kind == "llm_call":
                lines.append(
                    f"{head} role={e.data.get('role', 'n/a')} "
                    f"ms={e.data.get('ms')} tokens={e.data.get('tokens')}"
                )
            elif e.kind == "tool_call":
                lines.append(
                    f"{head} {e.data.get('tool')}({e.data.get('args')}) "
                    f"ms={e.data.get('ms')} -> {e.data.get('ok')}"
                )
            elif e.kind == "step":
                lines.append(f"{head} {e.data.get('n')}/{e.data.get('max_steps')} {e.data.get('summary', '')}")
            elif e.kind == "subagent":
                lines.append(
                    f"{head} spawn={e.data.get('index')} "
                    f"subtask={e.data.get('subtask', '')[:60]}"
                )
            elif e.kind == "finish":
                lines.append(f"{head} strategy={e.data.get('strategy')} answer_chars={e.data.get('answer_chars')}")
            else:
                lines.append(f"{head} {e.data}")
        return "\n".join(lines)

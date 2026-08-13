"""Router: Policy -> Execution Strategy dispatch.

Validates that tools required by the policy exist in the registry (v1:
advisory — ReAct resolves tools at call time anyway) and hands the task
to the executor selected by the policy.
"""

from __future__ import annotations

from typing import Any

from executors import Answer
from executors.direct import DirectExecutor
from executors.react import ReactExecutor
from executors.subagent import SubagentExecutor

EXECUTORS: dict[str, Any] = {
    DirectExecutor.strategy: DirectExecutor(),
    ReactExecutor.strategy: ReactExecutor(),
    SubagentExecutor.strategy: SubagentExecutor(),
}


class Router:
    def __init__(self, executors: dict[str, Any] | None = None) -> None:
        self.executors = executors or EXECUTORS

    def route(self, task: str, policy, llm: Any, trace: Any) -> Answer:
        strategy = policy.strategy
        if strategy not in self.executors:
            raise ValueError(f"no executor registered for strategy {strategy!r}")

        missing = [t for t in policy.tools_needed if not has_tool(t)]
        if missing:
            trace.record("dispatch", executor=strategy, warning=f"policy asked for unknown tools: {missing}")

        trace.record("dispatch", executor=strategy)
        executor = self.executors[strategy]
        return executor.execute(task, llm, trace)


def has_tool(name: str) -> bool:
    import tools

    return name in tools.TOOLS

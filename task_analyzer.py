"""Policy layer: Task -> Policy.

Two implementations of the same interface:
  - LLMPolicy: asks the model to classify the task (strategy, complexity,
    tools, reasoning). Most accurate, costs one LLM call.
  - RulePolicy: keyword heuristics. Zero cost, used as fallback when the
    LLM is unavailable or classification fails.

A `HybridPolicy` tries LLM first and falls back to rules — that is the
default used by the pipeline (v1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

STRATEGIES = ("direct", "react", "subagent")

POLICY_PROMPT = f"""You are the policy router of an agent runtime. Analyze the task and choose an
execution strategy.

Strategies:
- direct: answer from model knowledge alone; no tools, single call.
- react: needs tools (calculator, search) or multi-step reasoning with tool use.
- subagent: task is decomposable into independent subtasks, long, or parallel-friendly.

Return ONLY JSON:
{{"strategy": "<direct|react|subagent>", "complexity": "<low|medium|high>",
 "tools_needed": ["<calculator|search|...>"], "reasoning": "<one short sentence>"}}
"""

_MATH_RE = re.compile(r"\b(compute|calculate|calc|math|sum|solve|formula|(\d+)\s*[+\-*/^%])", re.I)
_SEARCH_RE = re.compile(r"\b(search|find|look up|what is|who is|when|where|about|facts?|corpus)\b", re.I)
_SUBAGENT_RE = re.compile(
    r"\b(compare|summarize (this |the )?(long )?|multi-step|parallel|for each|"
    r"all of the|every one|list (all|every)|write (a |an |the )?(report|analysis)|plan)\b",
    re.I,
)


@dataclass
class Policy:
    strategy: str
    complexity: str
    tools_needed: list[str] = field(default_factory=list)
    reasoning: str = ""
    source: str = ""  # "llm" | "rule" | "fallback"

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "complexity": self.complexity,
            "tools_needed": self.tools_needed,
            "reasoning": self.reasoning,
            "source": self.source,
        }


class RulePolicy:
    """Keyword heuristics. Deterministic and cheap."""

    def analyze(self, task: str) -> Policy:
        math = bool(_MATH_RE.search(task))
        search = bool(_SEARCH_RE.search(task))
        sub = bool(_SUBAGENT_RE.search(task))

        if sub:
            return Policy("subagent", "high", reasoning="decomposable/aggregate pattern detected", source="rule")
        if math and search:
            return Policy("react", "medium", ["calculator", "search"], "math + search patterns detected", source="rule")
        if math:
            return Policy("react", "medium", ["calculator"], "math pattern detected", source="rule")
        if search:
            return Policy("react", "low", ["search"], "search pattern detected", source="rule")
        return Policy("direct", "low", reasoning="no tool pattern detected", source="rule")


class LLMPolicy:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def analyze(self, task: str) -> Policy:
        parsed, meta = self.llm.chat_json(
            [
                {"role": "system", "content": POLICY_PROMPT},
                {"role": "user", "content": task},
            ],
            max_tokens=200,
        )
        strategy = parsed.get("strategy")
        if strategy not in STRATEGIES:
            strategy = "direct"
        complexity = parsed.get("complexity", "low")
        if complexity not in ("low", "medium", "high"):
            complexity = "low"
        tools = parsed.get("tools_needed")
        if not isinstance(tools, list):
            tools = []
        return Policy(
            strategy=strategy,
            complexity=complexity,
            tools_needed=[str(t) for t in tools],
            reasoning=str(parsed.get("reasoning", "")),
            source="llm",
        )


class HybridPolicy:
    """LLM first, rule fallback on any failure (no key, bad JSON, etc.)."""

    def __init__(self, llm: Any) -> None:
        self.llm_policy = LLMPolicy(llm)
        self.rule_policy = RulePolicy()

    def analyze(self, task: str) -> Policy:
        try:
            policy = self.llm_policy.analyze(task)
        except Exception:
            policy = self.rule_policy.analyze(task)
            policy.source = "fallback"
        return policy

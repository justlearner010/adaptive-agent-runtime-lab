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

# Policy prompt variants (experiment A: definition engineering).
# p0 = baseline (one-line definitions), p1 = operationalized + examples,
# p2 = data-grounded (encoded from EXPERIMENT-002 optimal labels).
PROMPT_VARIANTS: dict[str, str] = {
    "p0": f"""You are the policy router of an agent runtime. Analyze the task and choose an
execution strategy.

Strategies:
- direct: answer from model knowledge alone; no tools, single call.
- react: needs tools (calculator, search) or multi-step reasoning with tool use.
- subagent: task is decomposable into independent subtasks, long, or parallel-friendly.

Return ONLY JSON:
{{"strategy": "<direct|react|subagent>", "complexity": "<low|medium|high>",
 "tools_needed": ["<calculator|search|...>"], "reasoning": "<one short sentence>",
 "confidence": <0.0-1.0>}}
""",
    "p1": f"""You are the policy router of an agent runtime. Choose ONE execution strategy.

Strategies:
- direct: single answer from model knowledge, no tools.
  Use for: simple facts, common-sense Q&A, easy arithmetic, short comparisons/summaries.
  Examples: "what is the capital of france", "what is 2 + 2", "compare react and subagent"
- react: reason -> call a tool -> observe -> repeat until a final answer.
  Use for: queries about the local corpus (the model does not know it), or exact
  calculation that one-shot answers get wrong (large numbers, multi-step arithmetic).
  Examples: "search the corpus for compaction", "calculate 1234567 * 9876543 + 55555"
- subagent: split into independent subtasks with isolated contexts, then synthesize.
  Use for: long multi-topic reports, tasks with many independent parts.
  Example: "write a report on react, compaction, and subagent strategies"

Return ONLY JSON:
{{"strategy": "<direct|react|subagent>", "complexity": "<low|medium|high>",
 "tools_needed": ["<calculator|search|...>"], "reasoning": "<one short sentence>",
 "confidence": <0.0-1.0>}}
""",
    "p2": f"""You are the policy router of an agent runtime. Choose ONE execution strategy.

Empirical rule from benchmark data (strategy that is correct AND cheapest):
- direct is optimal by default: facts, common knowledge, easy/moderate arithmetic,
  comparisons, summaries, even multi-topic reports, all succeed in one call.
  Examples: "what is the capital of france", "what is 2 + 2",
            "compare react and subagent", "write a report on react and compaction"
- react is optimal ONLY when the task needs information outside model knowledge
  (corpus queries) or exact arithmetic one-shot answers get wrong (large products,
  long chains). When in doubt, prefer direct.
  Examples: "search the corpus for compaction", "calculate 1234567 * 9876543"
- subagent is optimal ONLY when the task clearly exceeds one-shot capacity and is
  naturally parallel; in practice almost nothing qualifies.

Return ONLY JSON:
{{"strategy": "<direct|react|subagent>", "complexity": "<low|medium|high>",
 "tools_needed": ["<calculator|search|...>"], "reasoning": "<one short sentence>",
 "confidence": <0.0-1.0>}}
""",
}

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
    confidence: float | None = None  # LLM self-reported confidence (0-1), v2

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "complexity": self.complexity,
            "tools_needed": self.tools_needed,
            "reasoning": self.reasoning,
            "source": self.source,
            "confidence": self.confidence,
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
    def __init__(self, llm: Any, variant: str = "p0") -> None:
        self.llm = llm
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"unknown policy prompt variant {variant!r}")
        self.prompt = PROMPT_VARIANTS[variant]

    def analyze(self, task: str) -> Policy:
        parsed, meta = self.llm.chat_json(
            [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": task},
            ],
            max_tokens=1024,  # EXPERIMENT-003: 512 was tight; reasoning models need headroom
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
            confidence=_parse_confidence(parsed.get("confidence")),
        )


def _parse_confidence(raw: Any) -> float | None:
    """Parse a 0-1 confidence value; invalid/missing -> None."""
    try:
        confidence = float(raw)
    except (TypeError, ValueError):
        return None
    return confidence if 0.0 <= confidence <= 1.0 else None


class HybridPolicy:
    """LLM first, rule fallback on any failure (no key, bad JSON, etc.)."""

    def __init__(self, llm: Any, variant: str = "p0") -> None:
        self.llm_policy = LLMPolicy(llm, variant=variant)
        self.rule_policy = RulePolicy()

    def analyze(self, task: str) -> Policy:
        try:
            policy = self.llm_policy.analyze(task)
        except Exception:
            policy = self.rule_policy.analyze(task)
            policy.source = "fallback"
        return policy

"""Aggregate eval results into a markdown report.

Tables:
1. accuracy: category x strategy (correct %)
2. cost: category x strategy (avg llm calls / tokens / latency)
3. agreement: policy vs rule vs optimal strategy, plus unsolved tasks
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from eval.runner import STRATEGIES


def optimal_strategy(entry: dict[str, Any]) -> str | None:
    """Cheapest strategy among correct ones: fewest llm_calls, then fewest tokens."""
    correct = {s: r for s, r in entry["runs"].items() if r.get("correct")}
    if not correct:
        return None
    return min(correct, key=lambda s: (correct[s]["llm_calls"], correct[s]["tokens"]))


def _pct(part: int, total: int) -> str:
    return f"{part}/{total} ({part * 100 // total if total else 0}%)" if total else "-"


def accuracy_table(results: dict[str, Any]) -> list[str]:
    tasks = results["tasks"]
    categories = sorted({t["category"] for t in tasks})
    lines = ["## 正确率（按任务类别 x 策略）", "", "| category | " + " | ".join(STRATEGIES) + " | n |", "|---|---" * (len(STRATEGIES) + 1) + "|"]
    for cat in categories:
        cat_tasks = [t for t in tasks if t["category"] == cat]
        cells = []
        for strategy in STRATEGIES:
            ok = sum(1 for t in cat_tasks if t["runs"].get(strategy, {}).get("correct"))
            cells.append(_pct(ok, len(cat_tasks)))
        lines.append(f"| {cat} | {' | '.join(cells)} | {len(cat_tasks)} |")
    ok_all = sum(1 for t in tasks if any(r.get("correct") for r in t["runs"].values()))
    lines.append(f"| **overall** | **{ok_all}/{len(tasks)} tasks solved by at least one strategy** |")
    return lines


def cost_table(results: dict[str, Any]) -> list[str]:
    tasks = results["tasks"]
    categories = sorted({t["category"] for t in tasks})
    lines = ["## 平均成本（按任务类别 x 策略）", "", "| category | strategy | llm_calls | tokens | latency_ms |", "|---|---|---|---|---|"]
    for cat in categories:
        cat_tasks = [t for t in tasks if t["category"] == cat]
        for strategy in STRATEGIES:
            runs = [t["runs"].get(strategy) for t in cat_tasks if t["runs"].get(strategy)]
            if not runs:
                continue
            avg = lambda key: sum(r.get(key, 0) for r in runs) // len(runs)  # noqa: E731
            lines.append(f"| {cat} | {strategy} | {avg('llm_calls')} | {avg('tokens')} | {avg('latency_ms')} |")
    return lines


def agreement_table(results: dict[str, Any]) -> list[str]:
    tasks = results["tasks"]
    lines = ["## 策略层质量（vs 最优策略）", "", "| id | category | optimal | policy | rule |", "|---|---|---|---|---|"]
    agree = {"policy": 0, "rule": 0}
    total = {"policy": 0, "rule": 0}
    unsolved: list[str] = []
    for t in tasks:
        optimal = optimal_strategy(t)
        if optimal is None:
            unsolved.append(f"{t['id']} ({t['task'][:40]})")
            continue
        p_choice = t["policy"].get("strategy")
        r_choice = t["rule_policy"].get("strategy")
        lines.append(f"| {t['id']} | {t['category']} | {optimal} | {p_choice} | {r_choice} |")
        if p_choice is not None:
            total["policy"] += 1
            agree["policy"] += p_choice == optimal
        if r_choice is not None:
            total["rule"] += 1
            agree["rule"] += r_choice == optimal
    if total["policy"]:
        lines.append(f"\n**policy (Hybrid) 与最优一致率**: {agree['policy']}/{total['policy']}")
    if total["rule"]:
        lines.append(f"**rule 与最优一致率**: {agree['rule']}/{total['rule']}")
    if unsolved:
        lines.append("\n**三策略均未解出的任务（评测集问题）**:")
        lines.extend(f"- {u}" for u in unsolved)
    return lines


def render_report(results: dict[str, Any]) -> str:
    meta = results.get("meta")
    header = [f"# Eval Report", ""]
    if meta:
        header = [f"# Eval Report ({meta.get('ts', '')})", "", f"model: {meta.get('model')}", ""]
    sections = [header, accuracy_table(results), [""], cost_table(results), [""], agreement_table(results)]
    return "\n".join(line for section in sections for line in section)

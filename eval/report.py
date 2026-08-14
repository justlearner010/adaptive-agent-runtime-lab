"""Aggregate eval results into a markdown report.

Tables:
1. accuracy: category x strategy (correct samples / total)
2. cost: category x strategy (mean +- std of llm_calls / tokens / latency)
3. stability: per-task policy classification distribution + llm success rate
4. agreement: majority policy vs rule vs optimal, plus unsolved tasks
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from eval.runner import STRATEGIES


# --- sample aggregation (pure) ------------------------------------------------------

def correct_rate(run_entry: dict[str, Any]) -> float:
    samples = run_entry.get("samples") or []
    if not samples:
        return 0.0
    return sum(1 for s in samples if s.get("correct")) / len(samples)


def sample_mean(run_entry: dict[str, Any], key: str) -> float:
    samples = run_entry.get("samples") or []
    values = [s.get(key, 0) for s in samples]
    return statistics.mean(values) if values else 0.0


def sample_std(run_entry: dict[str, Any], key: str) -> float:
    samples = run_entry.get("samples") or []
    values = [s.get(key, 0) for s in samples]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def optimal_strategy(entry: dict[str, Any]) -> str | None:
    """Highest correct rate; ties broken by cheapest (llm_calls, then tokens)."""
    rates = {s: correct_rate(r) for s, r in entry["runs"].items()}
    if not rates:
        return None
    best = max(rates.values())
    candidates = [s for s, v in rates.items() if v == best and best > 0]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda s: (sample_mean(entry["runs"][s], "llm_calls"), sample_mean(entry["runs"][s], "tokens")),
    )


def policy_majority(entry: dict[str, Any], variant: str = "p0") -> str | None:
    samples = entry.get("policy", {}).get(variant, {}).get("samples") or []
    choices = [s.get("strategy") for s in samples if s.get("strategy") not in (None, "error")]
    if not choices:
        return None
    # majority vote, ties broken by STRATEGIES order for determinism
    counts = Counter(choices)
    best = max(counts.values())
    tied = [s for s in STRATEGIES if counts.get(s) == best]
    return tied[0] if tied else None


def policy_llm_success_rate(entry: dict[str, Any], variant: str = "p0") -> float | None:
    samples = entry.get("policy", {}).get(variant, {}).get("samples") or []
    if not samples:
        return None
    return sum(1 for s in samples if s.get("source") == "llm") / len(samples)


# --- tables ---------------------------------------------------------------------------

def _pct(ok: int, total: int) -> str:
    return f"{ok}/{total} ({ok * 100 // total if total else 0}%)" if total else "-"


def accuracy_table(results: dict[str, Any]) -> list[str]:
    tasks = results["tasks"]
    categories = sorted({t["category"] for t in tasks})
    lines = ["## 正确率（按任务类别 x 策略，含全部采样）", "", "| category | " + " | ".join(STRATEGIES) + " | n |", "|---|---" * (len(STRATEGIES) + 1) + "|"]
    for cat in categories:
        cat_tasks = [t for t in tasks if t["category"] == cat]
        cells = []
        for strategy in STRATEGIES:
            ok = sum(1 for t in cat_tasks for s in (t["runs"].get(strategy) or {}).get("samples", []) if s.get("correct"))
            total = sum(len((t["runs"].get(strategy) or {}).get("samples", [])) for t in cat_tasks)
            cells.append(_pct(ok, total))
        lines.append(f"| {cat} | {' | '.join(cells)} | {len(cat_tasks)} |")
    solved = sum(1 for t in tasks if any(correct_rate(r) > 0 for r in t["runs"].values()))
    lines.append(f"| **overall** | **{solved}/{len(tasks)} tasks solved by at least one strategy** |")
    return lines


def cost_table(results: dict[str, Any]) -> list[str]:
    tasks = results["tasks"]
    categories = sorted({t["category"] for t in tasks})
    lines = ["## 平均成本（mean ± std，按任务类别 x 策略）", "", "| category | strategy | llm_calls | tokens | latency_ms | spawns |", "|---|---|---|---|---|---|"]
    for cat in categories:
        cat_tasks = [t for t in tasks if t["category"] == cat]
        for strategy in STRATEGIES:
            runs = [t["runs"].get(strategy) for t in cat_tasks if t["runs"].get(strategy)]
            if not runs:
                continue
            cells = []
            for key in ("llm_calls", "tokens", "latency_ms", "spawns"):
                mean = statistics.mean(sample_mean(r, key) for r in runs)
                std = statistics.stdev([sample_mean(r, key) for r in runs]) if len(runs) > 1 else 0.0
                cells.append(f"{mean:.1f}±{std:.1f}")
            lines.append(f"| {cat} | {strategy} | {' | '.join(cells)} |")
    return lines


def stability_table(results: dict[str, Any], variant: str = "p0") -> list[str]:
    lines = [f"## Policy 稳定性（N 次分类，prompt 变体 {variant}）", "", "| id | category | 选择分布 | llm成功率 |", "|---|---|---|---|"]
    for t in results["tasks"]:
        samples = t.get("policy", {}).get(variant, {}).get("samples") or []
        choices = [s.get("strategy") for s in samples]
        dist = ", ".join(f"{s}:{choices.count(s)}" for s in STRATEGIES if choices.count(s))
        if not dist:
            dist = "error"
        success = policy_llm_success_rate(t, variant)
        lines.append(f"| {t['id']} | {t['category']} | {dist} | {f'{success*100:.0f}%' if success is not None else '-'} |")
    return lines


def variant_summary_table(results: dict[str, Any]) -> list[str]:
    """One row per prompt variant: llm success rate + majority-vote agreement."""
    variants = sorted({v for t in results["tasks"] for v in t.get("policy", {})})
    lines = ["## Prompt 变体对比（实验 A/B）", "", "| variant | 任务数 | llm成功率(均值) | 多数票一致率 |", "|---|---|---|---|"]
    for variant in variants:
        tasks = results["tasks"]
        successes = [policy_llm_success_rate(t, variant) for t in tasks]
        successes = [s for s in successes if s is not None]
        avg = sum(successes) / len(successes) if successes else 0.0
        agree = total = 0
        for t in tasks:
            optimal = optimal_strategy(t)
            if optimal is None:
                continue
            total += 1
            agree += policy_majority(t, variant) == optimal
        lines.append(f"| {variant} | {len(tasks)} | {avg*100:.0f}% | {agree}/{total} |")
    return lines


def agreement_table(results: dict[str, Any], variant: str = "p0") -> list[str]:
    tasks = results["tasks"]
    lines = [f"## 策略层质量（多数票 vs 最优，prompt 变体 {variant}）", "", "| id | category | optimal | policy(majority) | rule |", "|---|---|---|---|---|"]
    agree = {"policy": 0, "rule": 0}
    total = {"policy": 0, "rule": 0}
    unsolved: list[str] = []
    for t in tasks:
        optimal = optimal_strategy(t)
        if optimal is None:
            unsolved.append(f"{t['id']} ({t['task'][:40]})")
            continue
        p_choice = policy_majority(t, variant)
        r_choice = t.get("rule_policy", {}).get("strategy")
        lines.append(f"| {t['id']} | {t['category']} | {optimal} | {p_choice} | {r_choice} |")
        if p_choice is not None:
            total["policy"] += 1
            agree["policy"] += p_choice == optimal
        if r_choice is not None:
            total["rule"] += 1
            agree["rule"] += r_choice == optimal
    if total["policy"]:
        lines.append(f"\n**policy (Hybrid, 多数票) 与最优一致率**: {agree['policy']}/{total['policy']}")
    if total["rule"]:
        lines.append(f"**rule 与最优一致率**: {agree['rule']}/{total['rule']}")
    if unsolved:
        lines.append("\n**三策略均未解出的任务（评测集问题）**:")
        lines.extend(f"- {u}" for u in unsolved)
    return lines


def baseline_table(results: dict[str, Any]) -> list[str]:
    """Degenerate-baseline agreement rates, as a reference frame for the
    policy's majority-vote agreement (issue #19).

    With most tasks optimal=direct, "always-direct" alone reaches a high
    agreement; the policy's agreement is only meaningful relative to these
    baselines.
    """
    tasks = results["tasks"]
    total = sum(1 for t in tasks if optimal_strategy(t) is not None)
    lines = ["## 退化基线一致率（参照系：多数票一致率应显著高于它们）", "", "| 策略 | 一致率 |", "|---|---|"]
    for s in STRATEGIES:
        agree = sum(1 for t in tasks if optimal_strategy(t) == s)
        lines.append(f"| always-{s} | {agree}/{total} |")
    rule_agree = sum(
        1
        for t in tasks
        if optimal_strategy(t) is not None and t.get("rule_policy", {}).get("strategy") == optimal_strategy(t)
    )
    lines.append(f"| rule（现状兜底） | {rule_agree}/{total} |")
    return lines


def render_report(results: dict[str, Any]) -> str:
    variants = sorted({v for t in results["tasks"] for v in t.get("policy", {})})
    default = variants[0] if variants else "p0"
    sections = [
        ["# Eval Report", ""],
        accuracy_table(results),
        [""],
        cost_table(results),
        [""],
        variant_summary_table(results),
        [""],
        stability_table(results, variant=default),
        [""],
        agreement_table(results, variant=default),
        [""],
        baseline_table(results),
    ]
    return "\n".join(line for section in sections for line in section)

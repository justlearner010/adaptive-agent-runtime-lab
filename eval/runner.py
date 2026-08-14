"""Eval runner: force each strategy over each task N times, collect metrics.

Usage:
    python -m eval.runner --runs 3 --limit 6     # smoke: 6 tasks, 3 samples each
    python -m eval.runner --runs 5 --workers 4    # full run, parallel
    python -m eval.runner --category math --runs 5
    python -m eval.runner --strategies react,subagent --runs 3
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from executors import Answer
from llm import LLM
from main import run_pipeline
from task_analyzer import HybridPolicy, RulePolicy
from tools.search import DEFAULT_CORPUS
from trace import Trace

STRATEGIES = ["direct", "react", "subagent"]
RESULTS_DIR = Path(__file__).parent / "results"

# corpus texts used by the grounding check for search-category correctness
_CORPUS_TEXTS = [text for _, text in DEFAULT_CORPUS]


# --- correctness checkers (pure) -------------------------------------------------

def extract_number(text: str) -> float | None:
    """Last numeric token in a text, tolerant of thousand separators."""
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def math_correct(answer: str, expected: str) -> bool:
    """True if the expected value appears among the numbers in the answer.

    Using "any number matches" (instead of the last number) avoids false
    negatives from verification tails, e.g. "…16. Check: 16×7=112, 112−13=99"
    where the last number (99) is not the answer.
    """
    target = float(expected.replace(",", ""))
    for num in re.findall(r"[-+]?\d+(?:\.\d+)?", answer.replace(",", "")):
        if abs(float(num) - target) < 1e-6:
            return True
    return False


def keyword_correct(answer: str, terms: list[str]) -> bool:
    low = answer.lower()
    return all(t.lower() in low for t in terms)


def _ngrams(text: str, n: int = 5) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def search_grounded(answer: str, corpus_texts: list[str], terms: list[str]) -> bool:
    """True if the answer quotes corpus text for EVERY required term.

    A single n-gram from any document is not enough: multi-topic tasks
    (e.g. must_contain ["calculator", "compaction"]) must show grounded
    evidence for each requested fact, or a generic mention plus one quote
    would pass. Keyword hits from generic model knowledge (e.g. "calculator
    … arithmetic") must not count as a solved search task.
    """
    answer_grams = _ngrams(answer)
    for term in terms:
        docs = [doc for doc in corpus_texts if term.lower() in doc.lower()]
        if not any(answer_grams & _ngrams(doc) for doc in docs):
            return False
    return True


def is_correct(task: dict, answer: Answer, events: list[dict], strategy: str) -> bool:
    if task["category"] in ("math", "chain"):
        return math_correct(answer.text, task["expected"])
    if task["category"] == "search":
        # keywords + per-term grounding: the answer must cite the corpus
        # for every requested fact
        return keyword_correct(answer.text, task.get("must_contain", [])) and search_grounded(
            answer.text, _CORPUS_TEXTS, task.get("must_contain", [])
        )
    if task["category"] == "subagent":
        # judged on the answer alone; mechanism fidelity (>=2 spawns) is a
        # separate metric, not part of correctness (planner may reasonably
        # decompose into a single subtask for simple tasks)
        return bool(answer.text.strip()) and keyword_correct(answer.text, task.get("must_contain", []))
    return keyword_correct(answer.text, task.get("must_contain", []))


# --- metrics (pure) ----------------------------------------------------------------

def metrics_from_trace(events: list[dict]) -> dict[str, Any]:
    llm_events = [e for e in events if e["kind"] == "llm_call"]
    tool_events = [e for e in events if e["kind"] == "tool_call"]
    return {
        "llm_calls": len(llm_events),
        "tokens": sum(e["data"].get("tokens") or 0 for e in llm_events),
        "latency_ms": sum(e["data"].get("ms") or 0 for e in llm_events),
        "tool_calls": len(tool_events),
        "tool_failures": sum(1 for e in tool_events if not e["data"].get("ok")),
        "spawns": sum(1 for e in events if e["kind"] == "subagent"),
    }


# --- single run ----------------------------------------------------------------------

def run_once(llm: LLM, task: dict, strategy: str) -> dict[str, Any]:
    """One (task, strategy) execution. Returns sample metrics; never raises."""
    trace = Trace()
    start = time.monotonic()
    try:
        _, answer = run_pipeline(task["task"], llm, force_strategy=strategy, trace=trace)
        events = trace.to_dict()
        return {
            "correct": is_correct(task, answer, events, strategy),
            "answer": answer.text[:300],
            "wall_ms": round((time.monotonic() - start) * 1000),
            **metrics_from_trace(events),
        }
    except Exception as exc:  # noqa: BLE001 - a broken strategy must not kill the eval
        return {
            "correct": False,
            "error": str(exc)[:200],
            "wall_ms": round((time.monotonic() - start) * 1000),
            "llm_calls": 0,
            "tokens": 0,
            "latency_ms": 0,
            "tool_calls": 0,
            "tool_failures": 0,
            "spawns": 0,
        }


# --- runner --------------------------------------------------------------------------

def run_eval(
    llm: LLM,
    tasks: list[dict],
    strategies: list[str],
    runs: int = 1,
    workers: int = 1,
    policy_variants: list[str] | None = None,
    on_task: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run each (task, strategy) `runs` times, optionally in parallel.

    Result shape per task:
        runs[strategy] = {"samples": [sample, ...]}
        policy[variant] = {"samples": [{"strategy", "source"}, ...]}  # per prompt variant
        rule_policy = {"strategy": "rule"}
    """
    policy_variants = policy_variants or ["p0"]
    results: dict[str, Any] = {"tasks": []}
    jobs = [(task, strategy, i) for task in tasks for strategy in strategies for i in range(runs)]
    policy_jobs = [(task, variant, i) for task in tasks for variant in policy_variants for i in range(runs)]

    def _submit(executor: ThreadPoolExecutor) -> list[tuple[str, tuple, Any]]:
        # interleave strategy and policy jobs so both workloads share the pool
        # concurrently; submitting them as two sequential batches would keep
        # the elapsed time additive instead of max(strategy, policy)
        pending: list[tuple[str, tuple, Any]] = []
        n = max(len(jobs), len(policy_jobs))
        for i in range(n):
            if i < len(jobs):
                job = jobs[i]
                pending.append(("strategy", job, executor.submit(run_once, llm, job[0], job[1])))
            if i < len(policy_jobs):
                job = policy_jobs[i]
                pending.append(("policy", job, executor.submit(_classify, llm, job[0], job[1])))
        return pending

    if workers > 1 and (len(jobs) + len(policy_jobs)) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = _submit(pool)
            strategy_outcomes = [(job, fut.result()) for kind, job, fut in pending if kind == "strategy"]
            policy_outcomes = [(job, fut.result()) for kind, job, fut in pending if kind == "policy"]
    else:
        strategy_outcomes = [(job, run_once(llm, *job)) for job in jobs]
        policy_outcomes = [(job, _classify(llm, *job)) for job in policy_jobs]

    by_key: dict[tuple[str, str], list[dict]] = {}
    for (task, strategy, _), sample in strategy_outcomes:
        by_key.setdefault((task["id"], strategy), []).append(sample)

    policy_by_key: dict[tuple[str, str], list[dict]] = {}
    for (task, variant, _), sample in policy_outcomes:
        policy_by_key.setdefault((task["id"], variant), []).append(sample)

    for task in tasks:
        entry: dict[str, Any] = {
            "id": task["id"],
            "category": task["category"],
            "task": task["task"],
            "runs": {},
            "policy": {},
            "rule_policy": {},
        }
        for strategy in strategies:
            entry["runs"][strategy] = {"samples": by_key.get((task["id"], strategy), [])}

        entry["policy"] = {
            variant: {"samples": policy_by_key.get((task["id"], variant), [])} for variant in policy_variants
        }

        rule = RulePolicy().analyze(task["task"])
        entry["rule_policy"] = {"strategy": rule.strategy, "source": "rule"}
        results["tasks"].append(entry)
        if on_task:
            on_task(task)
    return results


def _classify(llm: Any, task: dict, variant: str) -> dict:
    """One policy classification sample; never raises."""
    try:
        policy = HybridPolicy(llm, variant=variant).analyze(task["task"])
        return {"strategy": policy.strategy, "source": policy.source, "confidence": policy.confidence}
    except Exception as exc:  # noqa: BLE001
        # keep the raw model output (chat_json embeds it) for failure diagnosis
        return {"strategy": "error", "source": "error", "error": str(exc)[:300]}


def load_tasks(category: str | None = None, limit: int | None = None) -> list[dict]:
    tasks = json.loads((Path(__file__).parent / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    if category:
        tasks = [t for t in tasks if t["category"] == category]
    if limit:
        tasks = tasks[:limit]
    return tasks


def save_results(results: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{stamp}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="run at most N tasks")
    parser.add_argument("--category", default=None, choices=["math", "search", "direct", "subagent", "chain", "longdoc"])
    parser.add_argument("--strategies", default=",".join(STRATEGIES), help="comma-separated strategies")
    parser.add_argument("--runs", type=int, default=1, help="samples per (task, strategy)")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker count")
    parser.add_argument("--policy-variants", default="p0", help="comma-separated prompt variants (p0,p1,p2)")
    args = parser.parse_args(argv)

    from env import load_dotenv

    load_dotenv()
    llm = LLM()
    tasks = load_tasks(category=args.category, limit=args.limit)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    total = len(tasks) * len(strategies) * args.runs
    policy_variants = [v.strip() for v in args.policy_variants.split(",") if v.strip()]
    print(f"running {len(tasks)} tasks x {len(strategies)} strategies x {args.runs} runs = {total} executions ...")
    results = run_eval(
        llm, tasks, strategies, runs=args.runs, workers=args.workers,
        policy_variants=policy_variants,
        on_task=lambda t: print(f"  done: {t['id']} ({t['category']})", flush=True),
    )
    path = save_results(results)
    print(f"results -> {path}")

    from eval.report import render_report

    print(render_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Eval runner: force each strategy over each task, collect metrics from trace.

Usage:
    python -m eval.runner --limit 6            # smoke: 6 tasks x 3 strategies
    python -m eval.runner                      # full run
    python -m eval.runner --tasks subagent     # only one category
    python -m eval.runner --strategies react,subagent
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from executors import Answer
from llm import LLM
from main import run_pipeline
from task_analyzer import HybridPolicy, RulePolicy
from trace import Trace

STRATEGIES = ["direct", "react", "subagent"]
RESULTS_DIR = Path(__file__).parent / "results"


# --- correctness checkers (pure) -------------------------------------------------

def extract_number(text: str) -> float | None:
    """Last numeric token in a text, tolerant of thousand separators."""
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def math_correct(answer: str, expected: str) -> bool:
    value = extract_number(answer)
    if value is None:
        return False
    return abs(value - float(expected.replace(",", ""))) < 1e-6


def keyword_correct(answer: str, terms: list[str]) -> bool:
    low = answer.lower()
    return all(t.lower() in low for t in terms)


def subagent_correct(answer: str, events: list[dict], terms: list[str] | None) -> bool:
    """Structural check: the subagent strategy really decomposed the task.

    Only meaningful for subagent-strategy runs; other strategies are judged
    on the answer alone (see is_correct).
    """
    spawns = [e for e in events if e["kind"] == "subagent"]
    if len(spawns) < 2 or not answer.strip():
        return False
    return keyword_correct(answer, terms) if terms else True


def is_correct(task: dict, answer: Answer, events: list[dict], strategy: str) -> bool:
    if task["category"] == "math":
        return math_correct(answer.text, task["expected"])
    if task["category"] == "subagent":
        # task solved iff the answer covers the required topics; the structural
        # check (>=2 spawns) applies only to the subagent strategy itself
        if strategy == "subagent":
            return subagent_correct(answer.text, events, task.get("must_contain"))
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


# --- runner --------------------------------------------------------------------------

def _empty_run(error: str) -> dict[str, Any]:
    return {
        "correct": False,
        "error": error[:200],
        "wall_ms": 0,
        "llm_calls": 0,
        "tokens": 0,
        "latency_ms": 0,
        "tool_calls": 0,
        "tool_failures": 0,
        "spawns": 0,
    }


def run_eval(
    llm: LLM,
    tasks: list[dict],
    strategies: list[str],
    on_task=None,
) -> dict[str, Any]:
    results: dict[str, Any] = {"tasks": []}
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
            trace = Trace()
            start = time.monotonic()
            try:
                _, answer = run_pipeline(task["task"], llm, force_strategy=strategy, trace=trace)
                events = trace.to_dict()
                entry["runs"][strategy] = {
                    "correct": is_correct(task, answer, events, strategy),
                    "answer": answer.text[:300],
                    "wall_ms": round((time.monotonic() - start) * 1000),
                    **metrics_from_trace(events),
                }
            except Exception as exc:  # noqa: BLE001 - a broken strategy must not kill the whole eval
                entry["runs"][strategy] = _empty_run(str(exc))

        try:
            policy = HybridPolicy(llm).analyze(task["task"])
            entry["policy"] = {"strategy": policy.strategy, "source": policy.source}
        except Exception as exc:  # noqa: BLE001
            entry["policy"] = {"strategy": "error", "source": "error", "error": str(exc)[:100]}

        rule = RulePolicy().analyze(task["task"])
        entry["rule_policy"] = {"strategy": rule.strategy, "source": "rule"}
        results["tasks"].append(entry)
        if on_task:
            on_task(task)
    return results


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
    parser.add_argument("--category", default=None, choices=["math", "search", "direct", "subagent"])
    parser.add_argument("--strategies", default=",".join(STRATEGIES), help="comma-separated strategies")
    args = parser.parse_args(argv)

    from env import load_dotenv

    load_dotenv()
    llm = LLM()
    tasks = load_tasks(category=args.category, limit=args.limit)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    print(f"running {len(tasks)} tasks x {len(strategies)} strategies ...")
    results = run_eval(llm, tasks, strategies, on_task=lambda t: print(f"  done: {t['id']} ({t['category']})", flush=True))
    path = save_results(results)
    print(f"results -> {path}")

    from eval.report import render_report

    print(render_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

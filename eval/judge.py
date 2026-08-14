"""LLM-as-judge: re-judge stored eval answers without re-running the eval (v2).

Reads a saved eval results JSON (`eval/results/*.json`), runs one judge LLM
call per stored answer for the keyword-based categories (search / direct /
subagent / longdoc), and reports keyword-vs-judge agreement, false positives
(keyword says correct, judge says wrong) and false negatives.

Numeric categories (math / chain) are deterministic and are NOT re-judged.

Usage:
    python -m eval.judge --input eval/results/FULL-N5-PHASE3.json --out eval/results/judged.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from llm import LLM

SKIP_CATEGORIES = {"math", "chain"}
DEFAULT_WORKERS = 4

JUDGE_PROMPT = """You are an evaluation judge. Decide whether the answer correctly and
completely solves the given task. Be strict: partial or hallucinated answers are wrong.

Task: {task}

Answer:
{answer}

Respond ONLY with JSON:
{{"correct": <true|false>, "reason": "<one short sentence>"}}
"""


def judge_one(llm: Any, task_text: str, answer_text: str) -> dict[str, Any]:
    """One judge call. Returns {"correct": bool, "reason": str}."""
    parsed, _meta = llm.chat_json(
        [
            {"role": "system", "content": JUDGE_PROMPT.format(task=task_text, answer=answer_text)},
            {"role": "user", "content": "Judge now."},
        ],
        max_tokens=800,  # reasoning models need headroom before emitting JSON
    )
    return {"correct": bool(parsed.get("correct")), "reason": str(parsed.get("reason", ""))}


def judge_results(
    llm: Any,
    results: dict[str, Any],
    workers: int = DEFAULT_WORKERS,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return a copy of `results` with a `judge` field per sample (keyword categories)."""
    judged = json.loads(json.dumps(results))  # deep copy
    jobs: list[tuple[dict, str, int]] = []
    for task in judged["tasks"]:
        if task["category"] in SKIP_CATEGORIES:
            continue
        for strategy in ("direct", "react", "subagent"):
            samples = task.get("runs", {}).get(strategy, {}).get("samples") or []
            for i, sample in enumerate(samples):
                if sample.get("answer"):
                    jobs.append((task, strategy, i))

    def _judge(job: tuple[dict, str, int]) -> tuple[dict, str, int, dict]:
        task, strategy, i = job
        sample = task["runs"][strategy]["samples"][i]
        return task, strategy, i, judge_one(llm, task["task"], sample["answer"])

    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(_judge, jobs))
    else:
        done = [_judge(job) for job in jobs]

    seen: set[str] = set()
    for task, strategy, i, verdict in done:
        task["runs"][strategy]["samples"][i]["judge"] = verdict
        key = f"{task['id']}"
        if key not in seen:
            seen.add(key)
            if on_progress:
                on_progress(task["id"])
    return judged


def _confusion(results: dict[str, Any]) -> dict[str, Any]:
    """keyword-correct vs judge-correct, by category and overall."""
    stats: dict[str, dict[str, int]] = {"overall": {"n": 0, "agree": 0, "fp": 0, "fn": 0}}
    for task in results["tasks"]:
        cat = task["category"]
        if cat in SKIP_CATEGORIES:
            continue
        stats.setdefault(cat, {"n": 0, "agree": 0, "fp": 0, "fn": 0})
        for strategy in ("direct", "react", "subagent"):
            for sample in task.get("runs", {}).get(strategy, {}).get("samples") or []:
                if "judge" not in sample:
                    continue
                kw = bool(sample.get("correct"))
                jd = bool(sample["judge"].get("correct"))
                for bucket in ("overall", cat):
                    b = stats[bucket]
                    b["n"] += 1
                    if kw == jd:
                        b["agree"] += 1
                    elif kw and not jd:
                        b["fp"] += 1
                    else:
                        b["fn"] += 1
    return stats


def _fmt(stats: dict[str, dict[str, int]]) -> list[str]:
    lines = ["## 判定器质量：keyword vs LLM-as-judge", "", "| category | n | 一致率 | 假阳性(keyword对/judge错) | 假阴性(keyword错/judge对) |", "|---|---|---|---|---|"]
    for cat, s in stats.items():
        lines.append(
            f"| {cat} | {s['n']} | {s['agree']}/{s['n']} ({s['agree']*100//s['n'] if s['n'] else 0}%) "
            f"| {s['fp']} | {s['fn']} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="saved eval results JSON")
    parser.add_argument("--out", default=None, help="output JSON with judge verdicts")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="max tasks to judge (smoke)")
    args = parser.parse_args(argv)

    from env import load_dotenv

    load_dotenv()
    llm = LLM()
    results = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        results["tasks"] = results["tasks"][: args.limit]

    start = time.monotonic()
    judged = judge_results(llm, results, workers=args.workers, on_progress=lambda t: print(f"  judged: {t}", flush=True))
    print(f"judged in {time.monotonic() - start:.0f}s")

    if args.out:
        Path(args.out).write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"judged results -> {args.out}")

    print("\n".join(_fmt(_confusion(judged))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Adaptive Agent Runtime Lab — CLI entry point.

Pipeline: Task -> Policy (task_analyzer) -> Execution Strategy (router).

Usage:
    python main.py "task text"
    python main.py "task" --model deepseek-chat --base-url https://api.deepseek.com/v1
    python main.py "task" --offline        # rule-based policy, no LLM classification
    python main.py "task" --json           # emit machine-readable trace + answer
"""

from __future__ import annotations

import argparse
import json
import sys

from env import load_dotenv
from llm import LLM, LLMError
from router import Router
from task_analyzer import HybridPolicy, RulePolicy
from trace import Trace


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="the task to run")
    parser.add_argument("--api-key", default=None, help="override OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None, help="override OPENAI_BASE_URL")
    parser.add_argument("--model", default=None, help="override OPENAI_MODEL")
    parser.add_argument("--offline", action="store_true", help="use rule-based policy only (no LLM classification)")
    parser.add_argument("--json", action="store_true", dest="json_out", help="emit trace + answer as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()  # optional: read OPENAI_* from project .env, env vars still win
    trace = Trace()

    try:
        llm = LLM(api_key=args.api_key, base_url=args.base_url, model=args.model)
    except LLMError as exc:
        if args.offline:
            llm = None
        else:
            print(f"error: {exc}", file=sys.stderr)
            print("hint: pass --offline to run with rule-based policy and no LLM calls.", file=sys.stderr)
            return 1

    # 1) Task -> Policy
    if llm is None or args.offline:
        policy = RulePolicy().analyze(args.task)
    else:
        policy = HybridPolicy(llm).analyze(args.task)
    trace.record("policy", **policy.as_dict())

    # 2) Policy -> Execution Strategy
    router = Router()
    if llm is None:
        if policy.strategy != "direct":
            trace.record("dispatch", executor="direct", warning=f"offline: {policy.strategy} downgraded to direct (no LLM)")
        from executors import Answer

        answer = Answer(
            text="(offline mode: no LLM configured; execution skipped. "
            "Policy decision is shown in the trace above.)",
            strategy=policy.strategy,
            steps=0,
        )
    else:
        answer = router.route(args.task, policy, llm, trace)

    # 3) report
    trace.record("finish", strategy=answer.strategy, answer_chars=len(answer.text))

    if args.json_out:
        payload = {
            "task": args.task,
            "policy": policy.as_dict(),
            "answer": answer.text,
            "steps": answer.steps,
            "tool_calls": answer.tool_calls,
            "trace": trace.to_dict(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"task: {args.task}")
        print(f"policy: strategy={policy.strategy} complexity={policy.complexity} "
              f"tools={policy.tools_needed} source={policy.source} ({policy.reasoning})")
        print("---")
        print(trace.report())
        print("---")
        print(f"final answer ({answer.strategy}, {answer.steps} steps):")
        print(answer.text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

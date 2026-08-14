"""Tests for the LLM-as-judge module (no real LLM calls)."""

from eval.judge import _confusion, judge_one, judge_results


class FakeJudge:
    """Scripted chat_json responses for judge_one."""

    def __init__(self, verdicts: list[dict]):
        self._queue = list(verdicts)
        self.calls = 0

    def chat_json(self, messages, max_tokens=None, retries=1, structured=True):
        self.calls += 1
        return self._queue.pop(0), {"ms": 1, "tokens": 5}


def test_judge_one_parses_verdict():
    llm = FakeJudge([{"correct": True, "reason": "matches"}] * 5)
    v = judge_one(llm, "task text", "answer text")
    assert v == {"correct": True, "reason": "matches"}


def test_judge_results_annotates_samples_and_skips_numeric():
    results = {
        "tasks": [
            {
                "id": "search-01", "category": "search", "task": "search x",
                "runs": {
                    "direct": {"samples": [{"correct": True, "answer": "the corpus says interleaves reasoning"}]},
                    "react": {"samples": []},
                    "subagent": {"samples": []},
                },
            },
            {
                "id": "math-01", "category": "math", "task": "calc", "expected": "48",
                "runs": {
                    "direct": {"samples": [{"correct": True, "answer": "48"}]},
                    "react": {"samples": []},
                    "subagent": {"samples": []},
                },
            },
        ]
    }
    llm = FakeJudge([{"correct": False, "reason": "hallucinated"}] * 5)
    judged = judge_results(llm, results, workers=1)
    # search sample annotated
    assert judged["tasks"][0]["runs"]["direct"]["samples"][0]["judge"]["correct"] is False
    # math sample untouched (numeric category skipped)
    assert "judge" not in judged["tasks"][1]["runs"]["direct"]["samples"][0]
    assert llm.calls == 1


def test_confusion_counts_fp_and_fn():
    results = {
        "tasks": [
            {
                "id": "s1", "category": "search", "task": "x",
                "runs": {
                    "direct": {"samples": [
                        {"correct": True, "judge": {"correct": True}},   # agree
                        {"correct": True, "judge": {"correct": False}},  # FP
                        {"correct": False, "judge": {"correct": True}},  # FN
                        {"correct": False, "judge": {"correct": False}},  # agree
                    ]},
                    "react": {"samples": []},
                    "subagent": {"samples": []},
                },
            },
        ]
    }
    stats = _confusion(results)
    assert stats["overall"] == {"n": 4, "agree": 2, "fp": 1, "fn": 1}
    assert stats["search"] == {"n": 4, "agree": 2, "fp": 1, "fn": 1}

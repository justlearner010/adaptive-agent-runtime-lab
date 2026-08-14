"""Tests for eval runner/report pure logic (no LLM calls)."""

from eval.report import agreement_table, optimal_strategy
from eval.runner import (
    extract_number,
    is_correct,
    keyword_correct,
    math_correct,
    metrics_from_trace,
    search_grounded,
)


# --- correctness checkers ---

def test_extract_number():
    assert extract_number("The answer is 1,219,325,411,7436.") == 12193254117436.0
    assert extract_number("42.") == 42.0
    assert extract_number("-7 is the value") == -7.0
    assert extract_number("no numbers here") is None


def test_math_correct():
    assert math_correct("The result is 1093.", "1093")
    assert math_correct("Answer: 48", "48")
    assert not math_correct("Answer: 47", "48")
    assert not math_correct("I don't know", "1093")
    # verification tails must not cause false negatives: the expected value
    # appears among the numbers even though the last number is not the answer
    assert math_correct("The number is **16**. Check: 16×7=112, 112−13=99", "16")
    assert math_correct("1234567 × 9876543 + 55555 = 12193254117436", "12193254117436")


def test_keyword_correct_case_insensitive():
    assert keyword_correct("ReAct Interleaves reasoning", ["interleaves", "reasoning"])
    assert not keyword_correct("react is fine", ["interleaves"])


def test_search_grounded():
    corpus = ["ReAct interleaves reasoning steps (Thought) and tool calls (Action) with observations."]
    # quoting the corpus -> grounded
    assert search_grounded(
        "ReAct interleaves reasoning steps (Thought) and tool calls", corpus, ["interleaves", "reasoning"]
    )
    # generic knowledge without corpus text -> not grounded
    assert not search_grounded("ReAct combines thinking and acting with tools", corpus, ["interleaves"])
    assert not search_grounded("", corpus, ["interleaves"])


def test_search_grounded_requires_each_term():
    corpus = [
        "The calculator tool evaluates arithmetic expressions safely via an AST whitelist.",
        "Compaction summarizes long conversations to keep the context window bounded.",
    ]
    both = (
        "The calculator tool evaluates arithmetic expressions. "
        "Compaction summarizes long conversations to keep the context window bounded."
    )
    assert search_grounded(both, corpus, ["calculator", "compaction"])
    # quoting only one requested doc while mentioning the other generically must fail
    partial = "The calculator tool evaluates arithmetic expressions. Compaction is useful."
    assert not search_grounded(partial, corpus, ["calculator", "compaction"])


def test_is_correct_by_category():
    from executors import Answer

    math_task = {"category": "math", "expected": "48"}
    assert is_correct(math_task, Answer(text="48", strategy="x"), [], "direct")

    chain_task = {"category": "chain", "expected": "59"}
    assert is_correct(chain_task, Answer(text="the change is 59 yuan", strategy="x"), [], "direct")

    search_task = {"category": "search", "must_contain": ["interleaves", "reasoning"]}
    quoting = "ReAct interleaves reasoning steps (Thought) and tool calls (Action) with observations."
    assert is_correct(search_task, Answer(text=quoting, strategy="react"), [], "react")
    # keyword hit from generic knowledge without corpus grounding -> NOT correct
    assert not is_correct(
        search_task,
        Answer(text="ReAct interleaves reasoning with acting", strategy="direct"),
        [],
        "direct",
    )

    direct_task = {"category": "direct", "must_contain": ["paris"]}
    assert is_correct(direct_task, Answer(text="paris", strategy="x"), [], "direct")

    sub_task = {"category": "subagent", "must_contain": ["react", "subagent"]}
    # judged on the answer alone, regardless of strategy; spawns is a separate metric
    assert is_correct(sub_task, Answer(text="react and subagent report", strategy="subagent"), [], "subagent")
    assert not is_correct(sub_task, Answer(text="", strategy="subagent"), [], "subagent")
    assert is_correct(sub_task, Answer(text="react and subagent report", strategy="direct"), [], "direct")
    assert not is_correct(sub_task, Answer(text="nothing relevant", strategy="direct"), [], "direct")


# --- metrics ---

def test_metrics_from_trace():
    events = [
        {"kind": "llm_call", "data": {"tokens": 100, "ms": 500}},
        {"kind": "tool_call", "data": {"ok": True, "ms": 3}},
        {"kind": "tool_call", "data": {"ok": False, "ms": 1}},
        {"kind": "llm_call", "data": {"tokens": 50, "ms": 300}},
        {"kind": "subagent", "data": {}},
        {"kind": "dispatch", "data": {}},
    ]
    m = metrics_from_trace(events)
    assert m == {"llm_calls": 2, "tokens": 150, "latency_ms": 800, "tool_calls": 2, "tool_failures": 1, "spawns": 1}


# --- run_eval (parallel policy classification, no real LLM) ---

def test_run_eval_parallel_includes_policy_and_rule():
    from eval.runner import run_eval

    class FakeLLM:
        def chat(self, messages, max_tokens=None, retries=3, response_format=None):
            return "Final Answer: 4", {"ms": 1, "tokens": 5}

        def chat_json(self, messages, max_tokens=None, retries=1, structured=True):
            return {"strategy": "direct", "complexity": "low", "tools_needed": [], "reasoning": "x"}, {"ms": 1, "tokens": 5}

    tasks = [
        {"id": "math-01", "category": "math", "task": "calculate 23 * 47 + 12", "expected": "1093"},
        {"id": "search-01", "category": "search", "task": "search the corpus for what react is", "must_contain": ["interleaves", "reasoning"]},
    ]
    results = run_eval(FakeLLM(), tasks, strategies=["direct", "react"], runs=2, workers=2, policy_variants=["p0", "p1"])

    assert len(results["tasks"]) == 2
    t0 = results["tasks"][0]
    assert len(t0["runs"]["direct"]["samples"]) == 2
    assert len(t0["runs"]["react"]["samples"]) == 2
    for variant in ("p0", "p1"):
        assert [s["strategy"] for s in t0["policy"][variant]["samples"]] == ["direct", "direct"]
    assert t0["rule_policy"]["strategy"] == "react"  # "calculate" triggers the math rule


# --- report aggregation ---

def _samples(*corrects, calls=1, tokens=10):
    return {
        "samples": [
            {"correct": c, "llm_calls": calls, "tokens": tokens, "latency_ms": 100, "spawns": 0}
            for c in corrects
        ]
    }


def _entry(runs, policy_choices=("react", "react"), rule="react"):
    return {
        "id": "t1", "category": "math", "task": "x",
        "runs": runs,
        "policy": {
            "p0": {
                "samples": [
                    {"strategy": s, "source": ("error" if s == "error" else "llm")}
                    for s in policy_choices
                ]
            }
        },
        "rule_policy": {"strategy": rule, "source": "rule"},
    }


def test_correct_rate_and_means():
    from eval.report import correct_rate, sample_mean

    run = _samples(True, True, False, True)  # 3/4 correct
    assert correct_rate(run) == 0.75
    assert sample_mean(run, "llm_calls") == 1.0


def test_optimal_strategy_highest_rate_then_cheapest():
    from eval.report import optimal_strategy

    entry = _entry({
        "direct": _samples(True, True),                    # 100%, cheap
        "react": _samples(True, True, True),               # 100%, same rate
        "subagent": _samples(True, False),                 # 50%
    })
    # direct and react both 100% -> cheapest llm_calls wins (direct has calls=1)
    entry["runs"]["react"] = _samples(True, True, True, calls=3, tokens=900)
    entry["runs"]["direct"] = _samples(True, True, calls=1, tokens=10)
    assert optimal_strategy(entry) == "direct"


def test_optimal_strategy_tie_breaks_by_tokens():
    from eval.report import optimal_strategy

    entry = _entry({
        "direct": _samples(True, True, calls=2, tokens=500),
        "react": _samples(True, True, calls=2, tokens=300),
        "subagent": _samples(False, False),
    })
    assert optimal_strategy(entry) == "react"


def test_optimal_strategy_none_when_unsolved():
    from eval.report import optimal_strategy

    entry = _entry({"direct": _samples(False, False), "react": _samples(False, False), "subagent": _samples(False, False)})
    assert optimal_strategy(entry) is None


def test_policy_majority_and_llm_success():
    from eval.report import policy_llm_success_rate, policy_majority

    entry = _entry({}, policy_choices=("direct", "react", "react", "error"))
    assert policy_majority(entry) == "react"
    assert policy_llm_success_rate(entry) == 0.75  # 3 of 4 samples have source=llm
    assert policy_majority(entry, variant="p0") == "react"


def test_variant_summary_table():
    from eval.report import variant_summary_table

    # two variants: p0 agrees, p1 disagrees on the same optimal
    entry = {
        "id": "t1", "category": "math", "task": "x",
        "runs": {
            "direct": _samples(True, True, calls=1, tokens=1),
            "react": _samples(True, True, calls=2, tokens=2),
            "subagent": _samples(False, False),
        },
        "policy": {
            "p0": {"samples": [{"strategy": "direct", "source": "llm"}, {"strategy": "direct", "source": "llm"}]},
            "p1": {"samples": [{"strategy": "react", "source": "llm"}, {"strategy": "react", "source": "llm"}]},
        },
        "rule_policy": {"strategy": "react", "source": "rule"},
    }
    text = "\n".join(variant_summary_table({"tasks": [entry]}))
    assert "| p0 | 1 | 100% | 1/1 |" in text
    assert "| p1 | 1 | 100% | 0/1 |" in text


def test_agreement_table_counts():
    from eval.report import agreement_table

    results = {"tasks": [
        # t1: optimal=direct (both correct, direct cheaper); policy majority=direct -> agree
        _entry({"direct": _samples(True, True, calls=1, tokens=1),
                "react": _samples(True, True, calls=2, tokens=2),
                "subagent": _samples(False, False)}, policy_choices=("direct", "direct"), rule="direct"),
        # t2: optimal=react (react cheaper); policy majority=direct -> disagree; rule=direct -> disagree
        _entry({"direct": _samples(True, True, calls=2, tokens=2),
                "react": _samples(True, True, calls=1, tokens=1),
                "subagent": _samples(False, False)}, policy_choices=("direct", "direct"), rule="direct"),
        # t3: unsolved
        _entry({"direct": _samples(False, False), "react": _samples(False, False), "subagent": _samples(False, False)}),
    ]}
    text = "\n".join(agreement_table(results))
    assert "policy (Hybrid, 多数票) 与最优一致率**: 1/2" in text
    assert "rule 与最优一致率**: 1/2" in text
    assert "评测集问题" in text

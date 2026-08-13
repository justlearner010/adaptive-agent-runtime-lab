"""Tests for eval runner/report pure logic (no LLM calls)."""

from eval.report import agreement_table, optimal_strategy
from eval.runner import (
    extract_number,
    is_correct,
    keyword_correct,
    math_correct,
    metrics_from_trace,
    subagent_correct,
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


def test_keyword_correct_case_insensitive():
    assert keyword_correct("ReAct Interleaves reasoning", ["interleaves", "reasoning"])
    assert not keyword_correct("react is fine", ["interleaves"])


def test_subagent_correct_requires_spawns():
    events = [
        {"kind": "subagent", "data": {}},
        {"kind": "subagent", "data": {}},
    ]
    assert subagent_correct("compare direct and subagent", events, ["direct", "subagent"])
    assert not subagent_correct("compare", events, ["direct", "subagent"])
    assert not subagent_correct("compare direct and subagent", [{"kind": "llm_call", "data": {}}], None)


def test_is_correct_by_category():
    from executors import Answer

    math_task = {"category": "math", "expected": "48"}
    assert is_correct(math_task, Answer(text="48", strategy="x"), [], "direct")

    chain_task = {"category": "chain", "expected": "59"}
    assert is_correct(chain_task, Answer(text="the change is 59 yuan", strategy="x"), [], "direct")

    search_task = {"category": "search", "must_contain": ["interleaves"]}
    assert is_correct(search_task, Answer(text="ReAct interleaves reasoning", strategy="x"), [], "react")

    direct_task = {"category": "direct", "must_contain": ["paris"]}
    assert is_correct(direct_task, Answer(text="paris", strategy="x"), [], "direct")

    sub_task = {"category": "subagent", "must_contain": ["react"]}
    events = [{"kind": "subagent", "data": {}}, {"kind": "subagent", "data": {}}]
    # subagent strategy run: needs >=2 spawns + keywords
    assert is_correct(sub_task, Answer(text="react and subagent", strategy="subagent"), events, "subagent")
    assert not is_correct(sub_task, Answer(text="react and subagent", strategy="subagent"), [], "subagent")
    # direct/react run on the same task: judged on the answer alone
    assert is_correct(sub_task, Answer(text="react and subagent", strategy="direct"), [], "direct")
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

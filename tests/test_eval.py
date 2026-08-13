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
    assert is_correct(math_task, Answer(text="48", strategy="x"), [])

    search_task = {"category": "search", "must_contain": ["interleaves"]}
    assert is_correct(search_task, Answer(text="ReAct interleaves reasoning", strategy="x"), [])

    direct_task = {"category": "direct", "must_contain": ["paris"]}
    assert is_correct(direct_task, Answer(text="paris", strategy="x"), [])

    sub_task = {"category": "subagent", "must_contain": ["react"]}
    events = [{"kind": "subagent", "data": {}}, {"kind": "subagent", "data": {}}]
    assert is_correct(sub_task, Answer(text="react and subagent", strategy="x"), events)


# --- metrics ---

def test_metrics_from_trace():
    events = [
        {"kind": "llm_call", "data": {"tokens": 100, "ms": 500}},
        {"kind": "tool_call", "data": {"ok": True, "ms": 3}},
        {"kind": "tool_call", "data": {"ok": False, "ms": 1}},
        {"kind": "llm_call", "data": {"tokens": 50, "ms": 300}},
        {"kind": "dispatch", "data": {}},
    ]
    m = metrics_from_trace(events)
    assert m == {"llm_calls": 2, "tokens": 150, "latency_ms": 800, "tool_calls": 2, "tool_failures": 1}


# --- report aggregation ---

def _entry(runs, policy="react", rule="react"):
    return {
        "id": "t1", "category": "math", "task": "x",
        "runs": runs,
        "policy": {"strategy": policy, "source": "llm"},
        "rule_policy": {"strategy": rule, "source": "rule"},
    }


def test_optimal_strategy_cheapest_correct():
    entry = _entry({
        "direct": {"correct": True, "llm_calls": 1, "tokens": 10},
        "react": {"correct": True, "llm_calls": 3, "tokens": 900},
        "subagent": {"correct": False, "llm_calls": 5, "tokens": 1200},
    })
    assert optimal_strategy(entry) == "direct"


def test_optimal_strategy_tie_breaks_by_tokens():
    entry = _entry({
        "direct": {"correct": True, "llm_calls": 2, "tokens": 500},
        "react": {"correct": True, "llm_calls": 2, "tokens": 300},
        "subagent": {"correct": False, "llm_calls": 1, "tokens": 100},
    })
    assert optimal_strategy(entry) == "react"


def test_optimal_strategy_none_when_unsolved():
    entry = _entry({
        "direct": {"correct": False, "llm_calls": 1, "tokens": 1},
        "react": {"correct": False, "llm_calls": 1, "tokens": 1},
        "subagent": {"correct": False, "llm_calls": 1, "tokens": 1},
    })
    assert optimal_strategy(entry) is None


def test_agreement_table_counts():
    results = {"tasks": [
        _entry({"direct": {"correct": True, "llm_calls": 1, "tokens": 1},
                "react": {"correct": False, "llm_calls": 2, "tokens": 2},
                "subagent": {"correct": False, "llm_calls": 3, "tokens": 3}}, policy="direct"),
        _entry({"direct": {"correct": True, "llm_calls": 1, "tokens": 1},
                "react": {"correct": True, "llm_calls": 2, "tokens": 2},
                "subagent": {"correct": False, "llm_calls": 3, "tokens": 3}}, policy="react", rule="direct"),
        _entry({"direct": {"correct": False, "llm_calls": 1, "tokens": 1},
                "react": {"correct": False, "llm_calls": 2, "tokens": 2},
                "subagent": {"correct": False, "llm_calls": 3, "tokens": 3}}),
    ]}
    text = "\n".join(agreement_table(results))
    assert "policy (Hybrid) 与最优一致率**: 1/2" in text
    assert "rule 与最优一致率**: 1/2" in text
    assert "评测集问题" in text

"""Tests for the policy layer (rule fallback) and router wiring."""

from task_analyzer import RulePolicy
from router import Router


def test_rule_policy_classification():
    policy = RulePolicy()
    assert policy.analyze("calculate 23 * 47").strategy == "react"
    assert policy.analyze("search the corpus for react").strategy == "react"
    assert policy.analyze("what is pi").strategy == "react"
    assert policy.analyze("compare react and subagent").strategy == "subagent"
    assert policy.analyze("hello there").strategy == "direct"


def test_router_has_all_strategies():
    router = Router()
    assert set(router.executors) == {"direct", "react", "subagent"}


def test_router_rejects_unknown_strategy():
    from types import SimpleNamespace

    router = Router()
    policy = SimpleNamespace(strategy="bogus", tools_needed=[])
    try:
        router.route("x", policy, None, None)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

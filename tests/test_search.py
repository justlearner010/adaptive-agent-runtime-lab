"""Tests for the search tool."""

from tools.search import SearchTool

CORPUS = [
    ("react", "ReAct interleaves reasoning and tool calls."),
    ("subagent", "A subagent runs in an isolated context."),
]


def test_hit_returns_matching_fact():
    tool = SearchTool(corpus=CORPUS)
    result = tool.run({"input": "what is react"})
    assert "ReAct interleaves" in result


def test_no_hit_returns_no_results():
    tool = SearchTool(corpus=CORPUS)
    assert tool.run({"input": "quantum computing"}) == "No results found."


def test_empty_query_is_error():
    tool = SearchTool(corpus=CORPUS)
    assert tool.run({"input": ""}).startswith("Error")


def test_via_registry_execute():
    from tools import execute

    result = execute("search", {"input": "subagent"})
    assert "isolated context" in result

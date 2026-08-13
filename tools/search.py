"""Search tool, v1 placeholder.

Runs against a small local corpus so the ReAct mechanism is demonstrable
fully offline. Swap `_fetch` for a real search API (Tavily / SerpAPI /
DuckDuckGo) in a later version without changing the tool interface.

Scoring: token-overlap + substring bonus, top-k by score.
"""

from __future__ import annotations

import re
from typing import Iterable

# Small built-in corpus of facts so the search tool has something to find.
DEFAULT_CORPUS: list[tuple[str, str]] = [
    ("pi", "pi is a coding agent that runs a terminal-based interactive agent runtime."),
    ("react", "ReAct interleaves reasoning steps (Thought) and tool calls (Action) with observations."),
    ("subagent", "A subagent runs in an isolated context and reports results back to its parent agent."),
    ("policy", "A policy maps a task description to an execution strategy."),
    ("calculator", "The calculator tool evaluates arithmetic expressions safely via an AST whitelist."),
    ("compaction", "Compaction summarizes long conversations to keep the context window bounded."),
    ("openai", "OpenAI-compatible endpoints allow swapping models by changing base_url and model id."),
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score(query_tokens: set[str], doc_text: str) -> float:
    doc_tokens = _tokens(doc_text)
    if not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(len(query_tokens), len(doc_tokens))


class SearchTool:
    name = "search"
    description = (
        "Search the knowledge corpus. input: the query string, "
        "e.g. {'input': 'what is react'}. Returns the top matching facts."
    )

    def __init__(self, corpus: Iterable[tuple[str, str]] | None = None) -> None:
        self.corpus: list[tuple[str, str]] = list(corpus or DEFAULT_CORPUS)

    def run(self, args: dict, top_k: int = 3) -> str:
        query = args.get("input", "")
        if not isinstance(query, str) or not query.strip():
            return "Error: input must be a non-empty query string"
        query_tokens = _tokens(query)
        if not query_tokens:
            return "Error: query has no searchable tokens"
        ranked = sorted(
            ((_score(query_tokens, text), title) for title, text in self.corpus),
            key=lambda item: item[0],
            reverse=True,
        )
        hits = [(title, score) for title, score in ranked if score > 0][:top_k]
        if not hits:
            return "No results found."
        return "\n".join(f"- {title}: {text}" for title, text in [(t, dict(self.corpus)[t]) for t, _ in hits])

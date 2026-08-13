"""Tool registry and the two v1 tools: calculator and search.

Tools implement a uniform interface:
    name: str
    description: str
    run(args: dict) -> str   # always returns a string for the model to read
"""

from __future__ import annotations

from . import calculator, search

TOOLS: dict[str, object] = {
    calculator.CalculatorTool.name: calculator.CalculatorTool(),
    search.SearchTool.name: search.SearchTool(),
}


def describe() -> str:
    """Markdown-ish listing for injecting into prompts."""
    return "\n".join(
        f"- {name}: {tool.description}" for name, tool in sorted(TOOLS.items())
    )


def execute(name: str, args: dict) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return f"Unknown tool: {name}. Available: {', '.join(sorted(TOOLS))}"
    return tool.run(args)

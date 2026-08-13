"""Tests for the safe calculator evaluator."""

import pytest

from tools.calculator import CalculatorError, safe_eval


def test_basic_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(3 + 5) * 2") == 16
    assert safe_eval("2 ** 10") == 1024
    assert safe_eval("7 // 2") == 3
    assert safe_eval("7 % 2") == 1


def test_functions():
    assert safe_eval("sqrt(16)") == 4
    assert safe_eval("max(1, 5, 3)") == 5
    assert safe_eval("round(3.14159, 2)") == 3.14


def test_unsafe_syntax_rejected():
    for expr in [
        "__import__('os')",
        "open('/etc/passwd')",
        "1 if True else 2",
        "[x for x in range(3)]",
        "(1).real",
        "a + 1",
        "1/0",
    ]:
        with pytest.raises(CalculatorError):
            safe_eval(expr)


def test_calculator_tool_interface():
    from tools import execute

    assert execute("calculator", {"input": "6 * 7"}) == "42"
    assert execute("calculator", {"input": "not math"}).startswith("Error")
    assert "Unknown tool" in execute("nope", {})

"""Safe arithmetic calculator tool.

Evaluates arithmetic expressions with an AST whitelist so arbitrary code
execution is not possible: no attribute access, no names beyond a fixed
function set, no imports.
"""

from __future__ import annotations

import ast
import math
import operator as op

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "pow": pow,
}

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_MAX_RESULT_LEN = 200


class CalculatorError(ValueError):
    pass


def safe_eval(expr: str) -> float | int:
    """Evaluate a numeric expression, or raise CalculatorError."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression: {exc.msg}") from exc

    def _eval(node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _ALLOWED_FUNCTIONS.get(node.func.id)
            if fn is None:
                raise CalculatorError(f"function not allowed: {node.func.id}")
            args = [_eval(a) for a in node.args]
            if node.keywords:
                raise CalculatorError("keyword arguments are not allowed")
            return fn(*args)
        raise CalculatorError(f"unsupported syntax: {type(node).__name__}")

    try:
        result = _eval(tree.body)
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        raise CalculatorError(str(exc)) from exc
    text = str(result)
    if len(text) > _MAX_RESULT_LEN:
        raise CalculatorError("result too large")
    return result


class CalculatorTool:
    name = "calculator"
    description = (
        "Evaluate a numeric expression. input: the expression string, "
        "e.g. {'input': '(3 + 5) * 2'}. Supports + - * / % ** // and "
        "sqrt/abs/round/floor/ceil/min/max/pow."
    )

    def run(self, args: dict) -> str:
        expr = args.get("input", "")
        if not isinstance(expr, str) or not expr.strip():
            return "Error: input must be a non-empty expression string"
        try:
            return str(safe_eval(expr))
        except CalculatorError as exc:
            return f"Error: {exc}"

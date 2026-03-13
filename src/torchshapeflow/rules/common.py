from __future__ import annotations

import ast

from torchshapeflow.model import (
    ConstantDim,
    Dim,
    ExpressionDim,
    IntegerValue,
    ShapeTupleValue,
    render_dim,
)


def qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = qualified_name(node.value)
        if not base:
            return node.attr
        return f"{base}.{node.attr}"
    return ""


def is_name_or_attr(node: ast.AST, value: str) -> bool:
    return qualified_name(node).endswith(value)


def int_from_ast(node: ast.AST) -> int | None:
    """Evaluate a statically-known integer expression from an AST node.

    Supports integer literals, unary negation, and binary Add/Sub/Mult.
    Returns None for any expression that cannot be resolved at analysis time
    (e.g. variable references, division, or more complex expressions).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = int_from_ast(node.operand)
        if operand is not None:
            return -operand
    if isinstance(node, ast.BinOp):
        left = int_from_ast(node.left)
        right = int_from_ast(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


def dim_from_value(value: IntegerValue | Dim) -> Dim:
    """Convert an IntegerValue or Dim to a Dim, using ExpressionDim("unknown") for None."""
    if isinstance(value, IntegerValue):
        if value.value is None:
            return ExpressionDim("unknown")
        return ConstantDim(value.value)
    return value


def tuple_index(value: ShapeTupleValue, index: int) -> Dim | None:
    """Index into a ShapeTupleValue, supporting negative indices. Returns None if out of bounds."""
    if index < 0:
        index += len(value.dims)
    if index < 0 or index >= len(value.dims):
        return None
    return value.dims[index]


def render_dims(dims: tuple[Dim, ...]) -> str:
    return "[" + ", ".join(render_dim(dim) for dim in dims) + "]"

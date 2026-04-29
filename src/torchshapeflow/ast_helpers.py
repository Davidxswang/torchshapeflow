"""Pure AST/Dim helpers used by the analyzer.

These helpers do not reference the analyzer's mutable evaluation state
(``_eval_expr``, ``ModuleContext`` mutation, or the function environment)
and so live outside ``analyzer.py`` to keep that file focused on the
walker logic.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

from torchshapeflow.arithmetic import product_dim, quotient_dim, sum_dim
from torchshapeflow.model import (
    ConstantDim,
    Dim,
    ExpressionDim,
    SymbolicDim,
    TensorShape,
    TensorTupleValue,
    TensorValue,
    render_dim,
)
from torchshapeflow.rules import infer_split
from torchshapeflow.rules.common import int_from_ast

# ---------------------------------------------------------------------------
# Call-argument extraction (positional / keyword / pair / pool-stride)
# ---------------------------------------------------------------------------


def keyword_or_default(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def int_pair(
    node: ast.AST | None,
    default: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    if node is None:
        return default
    single = int_from_ast(node)
    if single is not None:
        return single, single
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        first = int_from_ast(node.elts[0])
        second = int_from_ast(node.elts[1])
        if first is not None and second is not None:
            return first, second
    return None


def pool_stride(call: ast.Call, kernel_size: tuple[int, int]) -> tuple[int, int] | None:
    """Return the effective stride for a pooling call.

    PyTorch pool layers default stride to kernel_size when the argument is absent.
    Checks positional arg[1] first, then the ``stride`` keyword.
    """
    stride_node: ast.AST | None = None
    if len(call.args) >= 2:
        stride_node = call.args[1]
    else:
        stride_node = keyword_or_default(call, "stride")
    if stride_node is None:
        return kernel_size
    return int_pair(stride_node)


def int_or_tuple(node: ast.expr) -> int | tuple[int, ...] | None:
    """Parse an AST node as an int or a tuple/list of ints (for movedim source/destination)."""
    v = int_from_ast(node)
    if v is not None:
        return v
    if isinstance(node, (ast.Tuple, ast.List)):
        parts: list[int] = []
        for elt in node.elts:
            i = int_from_ast(elt)
            if i is None:
                return None
            parts.append(i)
        return tuple(parts)
    return None


def keyword_int(node: ast.Call, name: str, default: int | None) -> int | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return int_from_ast(keyword.value)
    return default


def positional_int(
    args: Sequence[ast.expr],
    index: int,
    default: int | None,
) -> int | None:
    if index >= len(args):
        return default
    return int_from_ast(args[index])


def reduction_dim(node: ast.Call, arg_offset: int) -> int | tuple[int, ...] | None:
    """Extract the ``dim`` argument from a reduction call."""
    dim_node: ast.AST | None = None
    if len(node.args) > arg_offset:
        dim_node = node.args[arg_offset]
    else:
        for kw in node.keywords:
            if kw.arg == "dim":
                dim_node = kw.value
                break
    if dim_node is None:
        return None
    if isinstance(dim_node, ast.Tuple):
        ints = [int_from_ast(elt) for elt in dim_node.elts]
        if any(i is None for i in ints):
            return None
        return tuple(int(i) for i in ints if i is not None)
    return int_from_ast(dim_node)


def reduction_keepdim(node: ast.Call, positional_index: int) -> bool:
    """Extract the ``keepdim`` flag from a reduction call (keyword or positional bool)."""
    for kw in node.keywords:
        if kw.arg == "keepdim":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                return kw.value.value
    if len(node.args) > positional_index:
        arg = node.args[positional_index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, bool):
            return arg.value
    return False


def arange_length(node: ast.Call) -> int | None:
    """Return the number of elements in a ``torch.arange`` call if constant."""
    if len(node.args) == 1:
        return int_from_ast(node.args[0])
    if len(node.args) == 2:
        start = int_from_ast(node.args[0])
        end = int_from_ast(node.args[1])
        if start is not None and end is not None and end >= start:
            return end - start
    if len(node.args) == 3:
        start = int_from_ast(node.args[0])
        end = int_from_ast(node.args[1])
        step = int_from_ast(node.args[2])
        if start is not None and end is not None and step is not None and step > 0:
            return (end - start + step - 1) // step
    return None


# ---------------------------------------------------------------------------
# Dim / shape arithmetic over AST
# ---------------------------------------------------------------------------


def dim_binop(op: ast.operator, left: Dim | int, right: Dim | int) -> Dim | None:
    """Apply an arithmetic operator to two dims, returning a combined Dim."""
    ld = ConstantDim(left) if isinstance(left, int) else left
    rd = ConstantDim(right) if isinstance(right, int) else right
    if isinstance(op, ast.Mult):
        return product_dim((ld, rd))
    if isinstance(op, ast.Add):
        return sum_dim((ld, rd))
    if isinstance(op, ast.Sub):
        if isinstance(ld, ConstantDim) and isinstance(rd, ConstantDim):
            return ConstantDim(ld.value - rd.value)
        return ExpressionDim(f"({render_dim(ld)} - {render_dim(rd)})")
    if isinstance(op, ast.FloorDiv):
        return quotient_dim((ld,), (rd,))
    return None


def shapes_definitely_mismatch(declared: TensorShape, actual: TensorShape) -> bool:
    """Return True only when the shapes are provably incompatible."""
    if declared.rank != actual.rank:
        return True
    for d, a in zip(declared.dims, actual.dims, strict=True):
        if isinstance(d, ConstantDim) and isinstance(a, ConstantDim) and d.value != a.value:
            return True
        if isinstance(d, SymbolicDim) and isinstance(a, SymbolicDim) and d.name != a.name:
            return True
        if isinstance(d, ConstantDim) and isinstance(a, SymbolicDim):
            return True
        if isinstance(d, SymbolicDim) and isinstance(a, ConstantDim):
            return True
    return False


def infer_repeat_call(tensor: TensorValue, node: ast.Call) -> TensorValue:
    """Infer output shape of ``x.repeat(*repeats)``."""
    size_nodes: list[ast.expr] = list(node.args)
    if len(size_nodes) == 1 and isinstance(size_nodes[0], (ast.Tuple, ast.List)):
        size_nodes = list(size_nodes[0].elts)
    n = len(size_nodes)
    rank = tensor.rank
    if n < rank:
        return tensor
    padded: tuple[Dim, ...] = (ConstantDim(1),) * (n - rank) + tensor.shape.dims
    result_dims: list[Dim] = []
    for d, size_node in zip(padded, size_nodes, strict=True):
        repeat_val = int_from_ast(size_node)
        if repeat_val is not None:
            result_dims.append(product_dim((ConstantDim(repeat_val), d)))
        else:
            result_dims.append(ExpressionDim(f"{d}*?"))
    return TensorValue(TensorShape(tuple(result_dims)))


def split_from_call(tensor: TensorValue, node: ast.Call) -> TensorTupleValue | None:
    """Parse split arguments from a ``.split(size_or_sections, dim)`` call."""
    if not node.args:
        return None
    size_node = node.args[0]
    dim = positional_int(node.args, 1, None)
    if dim is None:
        dim = keyword_int(node, "dim", 0)
    if dim is None:
        dim = 0
    if isinstance(size_node, (ast.List, ast.Tuple)):
        sizes = [int_from_ast(e) for e in size_node.elts]
        if any(s is None for s in sizes):
            return None
        return infer_split(tensor, [s for s in sizes if s is not None], dim)
    split_size = int_from_ast(size_node)
    if split_size is not None:
        return infer_split(tensor, split_size, dim)
    return None


# ---------------------------------------------------------------------------
# Function-body shape predicates (for `tsf suggest`)
# ---------------------------------------------------------------------------


def contains_top_level_yield(body: list[ast.stmt]) -> bool:
    """True iff *body* contains a ``yield`` or ``yield from`` not inside a nested callable.

    The analyzer uses this to skip suggesting a return-tensor annotation on
    generator functions: when a function contains ``yield``,
    it returns a ``Generator[...]`` object, never the tensor the ``return``
    statement names (which becomes the ``StopIteration`` value). We must not
    propose a plain-tensor return annotation for generators.

    Walks the statement tree but does not descend into nested ``def``,
    ``async def``, or ``lambda`` bodies: a yield inside one of those makes the
    inner callable a generator, not the outer one.
    """
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def body_terminates_with_return(body: list[ast.stmt]) -> bool:
    """True iff *body* provably ends by returning a value.

    Recognizes:
    - A trailing ``return X`` (with value) at the end of the body.
    - A trailing ``raise`` (function exits without falling through to None).
    - A trailing ``if`` with ``else`` where every branch terminates.

    Loops, ``try``/``except``, ``match``, and bare ``return`` (no value) yield
    False — an honest "don't know" that keeps ``tsf suggest`` silent rather
    than asserting a shape contract the analyzer cannot prove.
    """
    if not body:
        return False
    last = body[-1]
    if isinstance(last, ast.Return) and last.value is not None:
        return True
    if isinstance(last, ast.Raise):
        return True
    if isinstance(last, ast.If) and last.orelse:
        return body_terminates_with_return(last.body) and body_terminates_with_return(last.orelse)
    return False


# ---------------------------------------------------------------------------
# Alias / TypeAlias targeting (for env merging and hover emission)
# ---------------------------------------------------------------------------


def alias_target_node(statement: ast.stmt) -> ast.Name | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            return target
        return None
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target
    alias_name = type_alias_name_node(statement)
    if alias_name is not None:
        return alias_name
    return None


def type_alias_name_node(statement: ast.stmt) -> ast.Name | None:
    type_alias_cls = getattr(ast, "TypeAlias", None)
    if type_alias_cls is None or not isinstance(statement, type_alias_cls):
        return None
    name = getattr(statement, "name", None)
    return name if isinstance(name, ast.Name) else None


def is_first_iteration_test(test: ast.AST, loop_var: str) -> bool:
    """True iff *test* compares ``loop_var`` to ``0`` (e.g. ``i == 0`` or ``0 == i``).

    Used to recognize first-iteration branches in loop-built ``Sequential``
    stacks where the first sub-module differs from later ones.
    """
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    return isinstance(test.ops[0], ast.Eq) and (
        (
            isinstance(left, ast.Name)
            and left.id == loop_var
            and isinstance(right, ast.Constant)
            and right.value == 0
        )
        or (
            isinstance(right, ast.Name)
            and right.id == loop_var
            and isinstance(left, ast.Constant)
            and left.value == 0
        )
    )

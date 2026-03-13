from __future__ import annotations

import ast

from torchshapeflow.model import ConstantDim, Dim, ShapeTupleValue, TensorShape, TensorValue
from torchshapeflow.rules.common import int_from_ast


def infer_subscript(
    value: TensorValue | ShapeTupleValue,
    node: ast.Subscript,
) -> TensorValue | ShapeTupleValue | Dim | None:
    """Infer the result of a subscript expression (``x[...]``).

    For **ShapeTupleValue** (e.g. ``x.shape[i]``): returns the Dim at the given
    integer index (supports negative indices).

    For **TensorValue** (e.g. ``x[0, :, None, ...]``): processes each slice element
    left-to-right according to these rules:

    * ``None`` / ``np.newaxis`` — inserts a new size-1 dimension.
    * ``Ellipsis`` (``...``) — consumes all remaining input dimensions unchanged.
    * ``slice`` (e.g. ``1:5``) — keeps the current dimension (size not tracked).
    * integer index — removes (consumes) the current dimension.

    Trailing unindexed dimensions are appended to the output.

    Returns:
        TensorValue for tensor subscripts, Dim for shape-tuple subscripts, or None if
        the index is out of bounds or the expression is unsupported.
    """
    if isinstance(value, ShapeTupleValue):
        index = int_from_ast(node.slice)
        if index is None:
            return None
        if index < 0:
            index += len(value.dims)
        if 0 <= index < len(value.dims):
            return value.dims[index]
        return None
    if not isinstance(value, TensorValue):
        return None
    slices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    dims = list(value.shape.dims)
    output: list[Dim] = []
    position = 0
    for slice_node in slices:
        if isinstance(slice_node, ast.Constant) and slice_node.value is None:
            output.append(ConstantDim(1))
            continue
        if isinstance(slice_node, ast.Constant) and slice_node.value is Ellipsis:
            output.extend(dims[position:])
            position = len(dims)
            continue
        if position >= len(dims):
            return None
        if isinstance(slice_node, ast.Slice):
            output.append(dims[position])
            position += 1
            continue
        if int_from_ast(slice_node) is not None:
            position += 1
            continue
        return None
    output.extend(dims[position:])
    return TensorValue(TensorShape(tuple(output)))

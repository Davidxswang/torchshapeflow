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
        # Slice of shape tuple (e.g. x.shape[-2:]) → ShapeTupleValue of the selected dims.
        if isinstance(node.slice, ast.Slice):
            n = len(value.dims)
            lower_val = 0 if node.slice.lower is None else int_from_ast(node.slice.lower)
            upper_val = n if node.slice.upper is None else int_from_ast(node.slice.upper)
            if lower_val is None or upper_val is None:
                return None
            if lower_val < 0:
                lower_val += n
            if upper_val < 0:
                upper_val += n
            lower_val = max(0, lower_val)
            upper_val = min(n, upper_val)
            return ShapeTupleValue(value.dims[lower_val:upper_val])
        # Single integer index (e.g. x.shape[0]) → a single Dim.
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
            step_is_one = slice_node.step is None or (
                isinstance(slice_node.step, ast.Constant) and slice_node.step.value == 1
            )
            lower_val = 0 if slice_node.lower is None else int_from_ast(slice_node.lower)
            upper_val = int_from_ast(slice_node.upper) if slice_node.upper is not None else None
            current_dim = dims[position]
            # Resolve open-ended upper bound when dim is constant
            if upper_val is None and isinstance(current_dim, ConstantDim):
                upper_val = current_dim.value
            # Resolve negative lower/upper bounds when dim is constant
            if lower_val is not None and lower_val < 0 and isinstance(current_dim, ConstantDim):
                lower_val = current_dim.value + lower_val
            if upper_val is not None and upper_val < 0 and isinstance(current_dim, ConstantDim):
                upper_val = current_dim.value + upper_val
            if (
                step_is_one
                and lower_val is not None
                and upper_val is not None
                and upper_val > lower_val
            ):
                output.append(ConstantDim(upper_val - lower_val))
            else:
                output.append(dims[position])
            position += 1
            continue
        if int_from_ast(slice_node) is not None:
            position += 1
            continue
        return None
    output.extend(dims[position:])
    return TensorValue(TensorShape(tuple(output)))

from __future__ import annotations

from torchshapeflow.arithmetic import normalize_index, product_dim, quotient_dim
from torchshapeflow.model import (
    ConstantDim,
    Dim,
    IntegerValue,
    ShapeTupleValue,
    TensorShape,
    TensorValue,
)


def infer_reshape(tensor: TensorValue, requested: tuple[Dim | int, ...]) -> TensorValue | None:
    """Infer shape after reshaping, supporting one -1 (inferred) dimension.

    Args:
        tensor: Input tensor. shape: (*input_dims)
        requested: Target shape as a tuple of Dim values or integers; at most one may be -1
            to request automatic inference.

    Returns:
        Output tensor with the requested shape. shape: (*requested_dims)
        None if more than one -1 is present, or if all dims are constant and the total
        element count does not match.
    """
    unknown_count = sum(1 for dim in requested if dim == -1)
    if unknown_count > 1:
        return None
    requested_dims = [
        dim if not isinstance(dim, int) else ConstantDim(dim) for dim in requested if dim != -1
    ]
    if unknown_count == 0:
        # For fully-constant shapes, validate that the element count is preserved.
        if all(isinstance(d, ConstantDim) for d in tensor.shape.dims) and all(
            isinstance(d, ConstantDim) for d in requested_dims
        ):
            if product_dim(tensor.shape.dims) != product_dim(tuple(requested_dims)):
                return None
        return TensorValue(TensorShape(tuple(requested_dims)))
    inferred = quotient_dim(tensor.shape.dims, tuple(requested_dims))
    if inferred is None:
        return None
    output_dims: list[Dim] = []
    for item in requested:
        if item == -1:
            output_dims.append(inferred)
        elif isinstance(item, int):
            output_dims.append(ConstantDim(item))
        else:
            output_dims.append(item)
    return TensorValue(TensorShape(tuple(output_dims)))


def infer_flatten(tensor: TensorValue, start_dim: int = 0, end_dim: int = -1) -> TensorValue | None:
    """Infer shape after flattening a contiguous range of dimensions.

    Args:
        tensor: Input tensor. shape: (*dims)
        start_dim: First dimension to flatten (supports negative indices). Default: 0.
        end_dim: Last dimension to flatten, inclusive (supports negative indices). Default: -1.

    Returns:
        Output tensor with the specified range collapsed into one dimension.
        shape: (*dims[:start], product(dims[start:end+1]), *dims[end+1:])
        None if either index is out of bounds or start_dim > end_dim.
    """
    start_index = normalize_index(start_dim, tensor.rank)
    end_index = normalize_index(end_dim, tensor.rank)
    if start_index is None or end_index is None or start_index > end_index:
        return None
    dims = list(tensor.shape.dims[:start_index])
    dims.append(product_dim(tensor.shape.dims[start_index : end_index + 1]))
    dims.extend(tensor.shape.dims[end_index + 1 :])
    return TensorValue(TensorShape(tuple(dims)))


def infer_squeeze(tensor: TensorValue, dim: int | None = None) -> TensorValue | None:
    """Infer shape after removing size-1 dimensions.

    Args:
        tensor: Input tensor. shape: (*dims)
        dim: If given, remove only that axis (supports negative indices); it must be size 1.
            If None, remove all size-1 axes.

    Returns:
        Output tensor with the relevant size-1 dimension(s) removed.
        None if dim is out of bounds or the specified dimension is not size 1.
    """
    dims = list(tensor.shape.dims)
    if dim is None:
        return TensorValue(TensorShape(tuple(value for value in dims if value != ConstantDim(1))))
    index = normalize_index(dim, tensor.rank)
    if index is None:
        return None
    if not isinstance(dims[index], ConstantDim):
        # Symbolic or unknown dim: cannot confirm it is size 1; return tensor unchanged.
        return tensor
    if dims[index] != ConstantDim(1):
        return None
    del dims[index]
    return TensorValue(TensorShape(tuple(dims)))


def infer_unsqueeze(tensor: TensorValue, dim: int) -> TensorValue | None:
    """Infer shape after inserting a size-1 dimension.

    Args:
        tensor: Input tensor. shape: (*dims)  (rank N)
        dim: Position at which to insert the new axis. Valid range: [-N-1, N].
            Negative indices count from the back of the *output* tensor, so -1 appends.
            The formula for negative dim is: index = rank + dim + 1.

    Returns:
        Output tensor with a new size-1 axis inserted. shape: (*dims with 1 at position index)
        None if dim is out of the valid range.
    """
    index = dim if dim >= 0 else tensor.rank + dim + 1
    if index < 0 or index > tensor.rank:
        return None
    dims = list(tensor.shape.dims)
    dims.insert(index, ConstantDim(1))
    return TensorValue(TensorShape(tuple(dims)))


def infer_size(
    tensor: TensorValue, dim: int | None = None
) -> ShapeTupleValue | IntegerValue | None:
    """Infer the result of tensor.size([dim]).

    Args:
        tensor: Input tensor. shape: (*dims)
        dim: If given, return the size of that axis (supports negative indices).
            If None, return the full shape tuple.

    Returns:
        ShapeTupleValue of all dims when dim is None.
        IntegerValue for the requested axis when it is a ConstantDim.
        None if dim is out of bounds or the axis is non-constant (symbolic/unknown).
    """
    if dim is None:
        return ShapeTupleValue(tensor.shape.dims)
    index = normalize_index(dim, tensor.rank)
    if index is None:
        return None
    selected = tensor.shape.dims[index]
    if isinstance(selected, ConstantDim):
        return IntegerValue(selected.value)
    # Symbolic/unknown dim: return IntegerValue(None) so callers can track the variable.
    return IntegerValue(None)

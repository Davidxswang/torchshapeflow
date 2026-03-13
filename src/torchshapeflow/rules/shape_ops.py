from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    Dim,
    IntegerValue,
    ShapeTupleValue,
    TensorShape,
    TensorValue,
    batch_matmul_shape,
    normalize_index,
    product_dim,
    quotient_dim,
    sum_dim,
)


def infer_permute(tensor: TensorValue, order: tuple[int, ...]) -> TensorValue | None:
    """Infer shape after permuting dimensions.

    Args:
        tensor: Input tensor. shape: (*dims)
        order: New ordering of axes (supports negative indices).

    Returns:
        Output tensor with dims reordered. shape: (dims[order[0]], dims[order[1]], ...)
        None if order length or uniqueness does not match the tensor rank, or an index is
        out of bounds.
    """
    if len(order) != tensor.rank or len(set(order)) != tensor.rank:
        return None
    normalized: list[int] = []
    for item in order:
        index = normalize_index(item, tensor.rank)
        if index is None:
            return None
        normalized.append(index)
    dims = tuple(tensor.shape.dims[index] for index in normalized)
    return TensorValue(TensorShape(dims))


def infer_transpose(tensor: TensorValue, first: int, second: int) -> TensorValue | None:
    """Infer shape after swapping two dimensions.

    Args:
        tensor: Input tensor. shape: (*dims)
        first: First axis to swap (supports negative indices).
        second: Second axis to swap (supports negative indices).

    Returns:
        Output tensor with the two axes swapped. shape: (*dims with first and second exchanged)
        None if either index is out of bounds.
    """
    first_index = normalize_index(first, tensor.rank)
    second_index = normalize_index(second, tensor.rank)
    if first_index is None or second_index is None:
        return None
    dims = list(tensor.shape.dims)
    dims[first_index], dims[second_index] = dims[second_index], dims[first_index]
    return TensorValue(TensorShape(tuple(dims)))


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
    if index is None or dims[index] != ConstantDim(1):
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
    return None


def infer_cat(values: tuple[TensorValue, ...], dim: int) -> TensorValue | None:
    """Infer shape after concatenating tensors along one dimension.

    Args:
        values: Tensors to concatenate; all must have equal rank and matching sizes on
            every axis except the concat axis. shape: each (*dims)
        dim: Axis to concatenate along (supports negative indices).

    Returns:
        Output tensor with the concat axis equal to the sum of input sizes on that axis.
        shape: (*dims with dims[index] = sum(t.shape.dims[index] for t in values))
        None if values is empty, dim is out of bounds, ranks differ, or any non-concat
        dimension is mismatched.
    """
    if not values:
        return None
    rank = values[0].rank
    index = normalize_index(dim, rank)
    if index is None:
        return None
    dims = list(values[0].shape.dims)
    concat_parts: list[Dim] = [values[0].shape.dims[index]]
    for value in values[1:]:
        if value.rank != rank:
            return None
        for dim_index, dim_val in enumerate(value.shape.dims):
            if dim_index == index:
                concat_parts.append(dim_val)
                continue
            if dim_val != dims[dim_index]:
                return None
    dims[index] = sum_dim(tuple(concat_parts))
    return TensorValue(TensorShape(tuple(dims)))


def infer_stack(values: tuple[TensorValue, ...], dim: int) -> TensorValue | None:
    """Infer shape after stacking tensors along a new dimension.

    Args:
        values: Tensors to stack; all must have identical shapes. shape: each (*dims)  (rank N)
        dim: Position of the new axis in the output. Valid range: [-N-1, N].
            Negative indices follow the same convention as infer_unsqueeze.

    Returns:
        Output tensor with a new axis of size len(values).
        shape: (*dims with ConstantDim(len(values)) inserted at position index)  (rank N+1)
        None if values is empty, shapes differ, or dim is out of the valid range.
    """
    if not values:
        return None
    reference = values[0].shape.dims
    if any(value.shape.dims != reference for value in values[1:]):
        return None
    index = dim if dim >= 0 else len(reference) + dim + 1
    if index < 0 or index > len(reference):
        return None
    dims = list(reference)
    dims.insert(index, ConstantDim(len(values)))
    return TensorValue(TensorShape(tuple(dims)))


def infer_matmul(left: TensorValue, right: TensorValue) -> TensorValue | None:
    """Infer shape after matrix multiplication (including batched matmul).

    Args:
        left: Left operand. shape: (*, M, K)  (rank >= 2)
        right: Right operand. shape: (*, K, N)  (rank >= 2)

    Returns:
        Output tensor with the inner dimension contracted.
        shape: (*, M, N) where * is the broadcast of the batch dimensions.
        None if either operand has rank < 2, the inner dimensions do not match,
        or the batch dimensions are incompatible.
    """
    shape = batch_matmul_shape(left.shape, right.shape)
    if shape is None:
        return None
    return TensorValue(shape)

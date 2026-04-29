from __future__ import annotations

from math import ceil

from torchshapeflow.arithmetic import normalize_index, sum_dim
from torchshapeflow.model import (
    ConstantDim,
    Dim,
    ExpressionDim,
    TensorShape,
    TensorTupleValue,
    TensorValue,
    render_dim,
)


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


def infer_chunk(tensor: TensorValue, n: int, dim: int) -> TensorTupleValue | None:
    """Infer shapes after ``x.chunk(n, dim)``.

    Splits the tensor into ``n`` equal-or-near-equal chunks along ``dim``.
    Each chunk has ``ceil(dim_size / n)`` on the split axis except the last,
    which may be smaller. When ``dim_size`` is a ``ConstantDim`` and exactly
    divisible by ``n``, all chunks are equal.

    Args:
        tensor: Input tensor of any rank.
        n: Number of chunks.
        dim: Axis to split along (negative indices supported).

    Returns:
        ``TensorTupleValue`` of ``n`` tensors, or ``None`` if ``dim`` is
        out of range.
    """
    norm_dim = normalize_index(dim, tensor.rank)
    if norm_dim is None:
        return None
    original = tensor.shape.dims[norm_dim]
    prefix = tensor.shape.dims[:norm_dim]
    suffix = tensor.shape.dims[norm_dim + 1 :]
    if isinstance(original, ConstantDim):
        total = original.value
        if total % n == 0:
            # All chunks equal.
            chunk_size = ConstantDim(total // n)
            chunk_shape = TensorShape(prefix + (chunk_size,) + suffix)
            return TensorTupleValue(tuple(TensorValue(chunk_shape) for _ in range(n)))
        # Non-divisible: first (n-1) chunks get ceil(total/n), last gets remainder.
        big = ceil(total / n)
        remainder = total - big * (n - 1)
        chunks: list[TensorValue] = []
        for i in range(n):
            size = big if i < n - 1 else remainder
            shape = TensorShape(prefix + (ConstantDim(size),) + suffix)
            chunks.append(TensorValue(shape))
        return TensorTupleValue(tuple(chunks))
    # Symbolic dim: express chunk size as a symbolic expression.
    chunk_dim: Dim = ExpressionDim(f"{render_dim(original)}//{n}")
    chunk_shape = TensorShape(prefix + (chunk_dim,) + suffix)
    return TensorTupleValue(tuple(TensorValue(chunk_shape) for _ in range(n)))


def infer_split(
    tensor: TensorValue,
    split_size: int | list[int],
    dim: int,
) -> TensorTupleValue | None:
    """Infer shapes after ``x.split(split_size_or_sections, dim)``.

    ``split_size`` may be:

    * An ``int`` — splits into equal-ish chunks of that size. Requires
      ``dim`` to be a ``ConstantDim``; returns ``None`` otherwise.
    * A ``list[int]`` — splits into exactly those sizes regardless of
      whether the dimension is known.

    Args:
        tensor: Input tensor of any rank.
        split_size: Uniform chunk size (int) or explicit per-chunk sizes
            (list of ints).
        dim: Axis to split along (negative indices supported).

    Returns:
        ``TensorTupleValue`` with one entry per chunk, or ``None`` if the
        split cannot be determined statically.
    """
    norm_dim = normalize_index(dim, tensor.rank)
    if norm_dim is None:
        return None
    prefix = tensor.shape.dims[:norm_dim]
    suffix = tensor.shape.dims[norm_dim + 1 :]

    if isinstance(split_size, list):
        # When the split dimension is constant, validate that sections sum to it.
        original = tensor.shape.dims[norm_dim]
        if isinstance(original, ConstantDim) and sum(split_size) != original.value:
            return None
        tensors = tuple(
            TensorValue(TensorShape(prefix + (ConstantDim(s),) + suffix)) for s in split_size
        )
        return TensorTupleValue(tensors)

    # Integer split_size: need a constant total to know chunk count.
    original = tensor.shape.dims[norm_dim]
    if not isinstance(original, ConstantDim):
        return None
    total = original.value
    chunks: list[TensorValue] = []
    remaining = total
    while remaining > 0:
        chunk = min(split_size, remaining)
        remaining -= chunk
        chunks.append(TensorValue(TensorShape(prefix + (ConstantDim(chunk),) + suffix)))
    return TensorTupleValue(tuple(chunks))

from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    Dim,
    IntegerValue,
    ShapeTupleValue,
    TensorShape,
    TensorTupleValue,
    TensorValue,
    UnknownDim,
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


def infer_reduction(
    tensor: TensorValue,
    dim: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
) -> TensorValue | None:
    """Infer shape after a reduction operation (sum, mean, max, min, etc.).

    Args:
        tensor: Input tensor of any rank.
        dim: Axis/axes to reduce over. ``None`` means reduce all axes (scalar result).
        keepdim: If True, reduced axes are kept as size-1 dimensions.

    Returns:
        Output TensorValue, or None if any axis index is out of range.
    """
    if dim is None:
        if keepdim:
            return TensorValue(TensorShape(tuple(ConstantDim(1) for _ in tensor.shape.dims)))
        return TensorValue(TensorShape(()))

    dims: tuple[int, ...]
    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = dim

    # Normalise and validate all axes first.
    norm_dims: list[int] = []
    for d in dims:
        nd = normalize_index(d, tensor.rank)
        if nd is None:
            return None
        norm_dims.append(nd)

    # Reject duplicate axis indices (PyTorch raises RuntimeError for duplicates).
    seen: set[int] = set()
    unique_dims: list[int] = []
    for nd in norm_dims:
        if nd in seen:
            return None
        seen.add(nd)
        unique_dims.append(nd)

    result: list[Dim] = []
    for i, dim_val in enumerate(tensor.shape.dims):
        if i in seen:
            if keepdim:
                result.append(ConstantDim(1))
        else:
            result.append(dim_val)
    return TensorValue(TensorShape(tuple(result)))


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
    if isinstance(original, ConstantDim) and original.value % n == 0:
        chunk_size: Dim = ConstantDim(original.value // n)
    else:
        chunk_size = UnknownDim("?")
    prefix = tensor.shape.dims[:norm_dim]
    suffix = tensor.shape.dims[norm_dim + 1 :]
    chunk_shape = TensorShape(prefix + (chunk_size,) + suffix)
    chunk_tv = TensorValue(chunk_shape)
    return TensorTupleValue(tuple(chunk_tv for _ in range(n)))


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


def infer_movedim(
    tensor: TensorValue,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
) -> TensorValue | None:
    """Infer shape after ``torch.movedim(x, source, destination)`` / ``x.movedim(...)``.

    Args:
        tensor: Input tensor. shape: (*dims) rank N
        source: Axis or axes to move (negative indices supported).
        destination: Target positions for those axes (negative indices supported).

    Returns:
        Output tensor with the specified axes moved. shape: (*reordered dims)
        None if any index is out of bounds or source/destination have different lengths.
    """
    rank = tensor.rank
    src = (source,) if isinstance(source, int) else source
    dst = (destination,) if isinstance(destination, int) else destination
    if len(src) != len(dst):
        return None
    norm_src = [normalize_index(s, rank) for s in src]
    norm_dst = [normalize_index(d, rank) for d in dst]
    if any(i is None for i in norm_src) or any(i is None for i in norm_dst):
        return None

    # After the None-check above, all entries are int.
    norm_src_int: list[int] = [i for i in norm_src if i is not None]
    norm_dst_int: list[int] = [i for i in norm_dst if i is not None]

    # Build output permutation: place moved axes at destination positions,
    # fill remaining slots with non-moved axes in original order.
    norm_src_set = set(norm_src_int)
    remaining = [i for i in range(rank) if i not in norm_src_set]
    perm: list[int | None] = [None] * rank
    for s, d in zip(norm_src_int, norm_dst_int, strict=True):
        perm[d] = s
    rem_iter = iter(remaining)
    for i in range(rank):
        if perm[i] is None:
            perm[i] = next(rem_iter)
    new_dims = tuple(tensor.shape.dims[p] for p in perm if p is not None)
    return TensorValue(TensorShape(new_dims))


def infer_mm(left: TensorValue, right: TensorValue) -> TensorValue | None:
    """Infer shape after ``torch.mm(x, y)`` (2-D matrix multiply, no batch).

    Args:
        left: Left matrix. shape: (M, K)  rank exactly 2
        right: Right matrix. shape: (K, N)  rank exactly 2

    Returns:
        Output tensor. shape: (M, N)
        None if either operand is not rank-2 or inner dims do not match.
    """
    from torchshapeflow.model import render_dim

    if left.rank != 2 or right.rank != 2:
        return None
    if render_dim(left.shape.dims[1]) != render_dim(right.shape.dims[0]):
        return None
    return TensorValue(TensorShape((left.shape.dims[0], right.shape.dims[1])))


def infer_interpolate(
    tensor: TensorValue,
    size: tuple[Dim, ...] | None,
    scale_factor: tuple[float, ...] | None,
) -> TensorValue | None:
    """Infer output shape of ``F.interpolate(input, size=..., scale_factor=...)``.

    Batch and channel dimensions (first two) are always preserved.
    Spatial dimensions (rank-2 trailing dims) are replaced by ``size`` when given,
    or scaled by ``scale_factor`` otherwise.

    Args:
        tensor: Input tensor. shape: (N, C, *spatial)  rank >= 3
        size: Target spatial sizes, one per spatial dim.
        scale_factor: Multiplier for each spatial dim; floats.

    Returns:
        Output TensorValue with new spatial dims, or None if rank < 3.
    """
    if tensor.rank < 3:
        return None
    batch_channel = tensor.shape.dims[:2]
    n_spatial = tensor.rank - 2
    if size is not None:
        if len(size) != n_spatial:
            return None
        return TensorValue(TensorShape(batch_channel + size))
    if scale_factor is not None:
        if len(scale_factor) != n_spatial:
            return None
        spatial = tensor.shape.dims[2:]
        new_spatial: list[Dim] = []
        for d, f in zip(spatial, scale_factor, strict=True):
            if isinstance(d, ConstantDim):
                new_spatial.append(ConstantDim(int(d.value * f)))
            else:
                new_spatial.append(UnknownDim("?"))
        return TensorValue(TensorShape(batch_channel + tuple(new_spatial)))
    return None


def infer_one_hot(tensor: TensorValue, num_classes: int) -> TensorValue:
    """Infer output shape of ``F.one_hot(tensor, num_classes)``.

    Args:
        tensor: Index tensor of any rank. shape: (*dims)
        num_classes: Number of classes; appended as a new trailing axis.

    Returns:
        Output tensor. shape: (*dims, num_classes)
    """
    return TensorValue(TensorShape(tensor.shape.dims + (ConstantDim(num_classes),)))


def infer_diagonal(
    tensor: TensorValue,
    offset: int,
    dim1: int,
    dim2: int,
) -> TensorValue | None:
    """Infer output shape of ``x.diagonal(offset, dim1, dim2)`` / ``torch.diagonal(...)``.

    Removes ``dim1`` and ``dim2`` from the shape and appends the diagonal length.

    Args:
        tensor: Input tensor. shape: (*dims)  rank >= 2
        offset: Diagonal offset (0 = main diagonal).
        dim1: First dimension of the 2-D sub-tensors.
        dim2: Second dimension of the 2-D sub-tensors.

    Returns:
        Output tensor. shape: (*remaining, min(d1, d2) - |offset|)
        None if indices are out of bounds or identical.
    """
    rank = tensor.rank
    n1 = normalize_index(dim1, rank)
    n2 = normalize_index(dim2, rank)
    if n1 is None or n2 is None or n1 == n2:
        return None
    d1 = tensor.shape.dims[n1]
    d2 = tensor.shape.dims[n2]
    if isinstance(d1, ConstantDim) and isinstance(d2, ConstantDim):
        diag_size = max(0, min(d1.value, d2.value) - abs(offset))
        diag_dim: Dim = ConstantDim(diag_size)
    else:
        diag_dim = UnknownDim("?")
    excluded = frozenset((n1, n2))
    remaining = tuple(d for i, d in enumerate(tensor.shape.dims) if i not in excluded)
    return TensorValue(TensorShape(remaining + (diag_dim,)))


def infer_index_select(
    tensor: TensorValue,
    dim: int,
    index_len: Dim,
) -> TensorValue | None:
    """Infer output shape of ``x.index_select(dim, index)`` / ``torch.index_select(...)``.

    Args:
        tensor: Input tensor. shape: (*dims)
        dim: Axis to select along.
        index_len: Number of indices selected (replaces ``dims[dim]``).

    Returns:
        Output tensor with ``dims[dim]`` replaced by ``index_len``, or None if out of bounds.
    """
    norm = normalize_index(dim, tensor.rank)
    if norm is None:
        return None
    new_dims = tensor.shape.dims[:norm] + (index_len,) + tensor.shape.dims[norm + 1 :]
    return TensorValue(TensorShape(new_dims))


def infer_topk(tensor: TensorValue, k: int, dim: int) -> TensorValue | None:
    """Infer values/indices shape of ``torch.topk(input, k, dim)``.

    Both the values and indices outputs have the same shape: the selected
    dimension becomes ``k``.

    Args:
        tensor: Input tensor. shape: (*dims)
        k: Number of top elements to keep.
        dim: Axis along which to find the top-k elements.

    Returns:
        Output tensor. shape: (*dims[:dim], k, *dims[dim+1:])
        None if the axis is out of bounds.
    """
    norm = normalize_index(dim, tensor.rank)
    if norm is None:
        return None
    new_dims = tensor.shape.dims[:norm] + (ConstantDim(k),) + tensor.shape.dims[norm + 1 :]
    return TensorValue(TensorShape(new_dims))


def infer_einsum(subscript: str, tensors: list[TensorValue]) -> TensorValue | None:
    """Infer output shape of ``torch.einsum(subscript, *tensors)``.

    Supports explicit-mode subscripts only (those containing ``->``) with
    single-character labels. Ellipsis and implicit mode are not yet supported.

    Args:
        subscript: Einstein summation string, e.g. ``"bik,bkj->bij"``.
        tensors: Input tensors in the same order as the subscript operands.

    Returns:
        Output ``TensorValue`` with dimensions inferred from the output labels,
        or ``None`` if parsing fails, label counts do not match tensor ranks,
        or the same label maps to conflicting constant dimensions.
    """
    subscript = subscript.replace(" ", "")
    if "->" not in subscript:
        return None  # implicit mode not yet supported
    lhs, rhs = subscript.split("->", 1)
    input_specs = lhs.split(",")
    if len(input_specs) != len(tensors):
        return None

    label_to_dim: dict[str, Dim] = {}
    for spec, tensor in zip(input_specs, tensors, strict=True):
        if len(spec) != tensor.rank:
            return None
        for label, dim in zip(spec, tensor.shape.dims, strict=True):
            existing = label_to_dim.get(label)
            if existing is not None:
                if (
                    isinstance(existing, ConstantDim)
                    and isinstance(dim, ConstantDim)
                    and existing.value != dim.value
                ):
                    return None  # contraction dimension mismatch
            else:
                label_to_dim[label] = dim

    output_dims: list[Dim] = []
    for label in rhs:
        if label not in label_to_dim:
            return None
        output_dims.append(label_to_dim[label])
    return TensorValue(TensorShape(tuple(output_dims)))

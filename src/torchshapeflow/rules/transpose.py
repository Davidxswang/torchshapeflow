from __future__ import annotations

from torchshapeflow.arithmetic import normalize_index
from torchshapeflow.model import TensorShape, TensorValue


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

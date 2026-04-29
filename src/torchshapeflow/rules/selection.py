from __future__ import annotations

from torchshapeflow.arithmetic import normalize_index, product_dim
from torchshapeflow.model import (
    ConstantDim,
    Dim,
    TensorShape,
    TensorValue,
    UnknownDim,
    render_dim,
)


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
    elif offset == 0 and render_dim(d1) == render_dim(d2):
        # Same symbolic dim and no offset: diagonal length equals the dim.
        diag_dim = d1
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


def infer_one_hot(tensor: TensorValue, num_classes: int) -> TensorValue:
    """Infer output shape of ``F.one_hot(tensor, num_classes)``.

    Args:
        tensor: Index tensor of any rank. shape: (*dims)
        num_classes: Number of classes; appended as a new trailing axis.

    Returns:
        Output tensor. shape: (*dims, num_classes)
    """
    return TensorValue(TensorShape(tensor.shape.dims + (ConstantDim(num_classes),)))


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
            elif f == int(f):
                # Integer scale factor: express as product (e.g. "2*H").
                new_spatial.append(product_dim((ConstantDim(int(f)), d)))
            else:
                new_spatial.append(UnknownDim("?"))
        return TensorValue(TensorShape(batch_channel + tuple(new_spatial)))
    return None

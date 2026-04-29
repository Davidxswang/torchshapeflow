from __future__ import annotations

from torchshapeflow.arithmetic import batch_matmul_shape, normalize_index
from torchshapeflow.model import (
    ConstantDim,
    Dim,
    TensorShape,
    TensorValue,
    render_dim,
)


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


def infer_mm(left: TensorValue, right: TensorValue) -> TensorValue | None:
    """Infer shape after ``torch.mm(x, y)`` (2-D matrix multiply, no batch).

    Args:
        left: Left matrix. shape: (M, K)  rank exactly 2
        right: Right matrix. shape: (K, N)  rank exactly 2

    Returns:
        Output tensor. shape: (M, N)
        None if either operand is not rank-2 or inner dims do not match.
    """
    if left.rank != 2 or right.rank != 2:
        return None
    if render_dim(left.shape.dims[1]) != render_dim(right.shape.dims[0]):
        return None
    return TensorValue(TensorShape((left.shape.dims[0], right.shape.dims[1])))


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
                if render_dim(existing) != render_dim(dim):
                    return None  # dimension mismatch (constant or symbolic)
            else:
                label_to_dim[label] = dim

    output_dims: list[Dim] = []
    for label in rhs:
        if label not in label_to_dim:
            return None
        output_dims.append(label_to_dim[label])
    return TensorValue(TensorShape(tuple(output_dims)))


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

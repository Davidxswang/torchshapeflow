from __future__ import annotations

from math import prod

from torchshapeflow.model import (
    ConstantDim,
    Dim,
    ExpressionDim,
    TensorShape,
    UnknownDim,
    render_dim,
)


def product_dim(dims: tuple[Dim, ...]) -> Dim:
    """Compute the product of a sequence of dimensions.

    All-constant inputs yield a ConstantDim. Mixed inputs yield an ExpressionDim whose
    string representation places the constant factor first (e.g. ``4*B*C``).

    Args:
        dims: Sequence of dimensions to multiply.

    Returns:
        ConstantDim with the numeric product if all dims are constant.
        ExpressionDim joining the parts with ``*`` otherwise.
    """
    constants: list[int] = []
    symbolic_parts: list[str] = []
    for dim in dims:
        if isinstance(dim, ConstantDim):
            constants.append(dim.value)
        else:
            symbolic_parts.append(render_dim(dim))
    if not symbolic_parts:
        return ConstantDim(prod(constants))
    constant_factor = prod(constants) if constants else 1
    if constant_factor != 1:
        symbolic_parts.insert(0, str(constant_factor))  # constant factor comes first
    if len(symbolic_parts) == 1:
        return ExpressionDim(symbolic_parts[0])
    return ExpressionDim("*".join(symbolic_parts))


def quotient_dim(numerator: tuple[Dim, ...], denominator: tuple[Dim, ...]) -> Dim | None:
    """Compute the quotient of two dimension-products.

    Used to infer the -1 dimension in a reshape. Cancels matching symbolic factors
    before computing the numeric quotient. Returns a ConstantDim when the result is
    a whole number after cancellation; an ExpressionDim when symbolic factors remain;
    or None when the constant remainder does not divide evenly (shape error).

    Args:
        numerator: Dimensions whose product forms the numerator.
        denominator: Dimensions whose product forms the denominator.

    Returns:
        ConstantDim if the result is a whole number, ExpressionDim if symbolic factors
        remain, or None if the constant remainder is not evenly divisible (invalid reshape).
    """
    # Cancel matching symbolic/expression factors by string comparison.
    # Note: "B*C" and "C*B" are treated as different (structural, not algebraic).
    num_list = list(numerator)
    den_list = list(denominator)
    for dim in list(den_list):
        for i, num_dim in enumerate(num_list):
            if render_dim(dim) == render_dim(num_dim):
                num_list.pop(i)
                den_list.remove(dim)
                break
    # After cancellation, check if all remaining are constant
    if all(isinstance(d, ConstantDim) for d in num_list + den_list):
        num = prod(d.value for d in num_list if isinstance(d, ConstantDim)) if num_list else 1
        den = prod(d.value for d in den_list if isinstance(d, ConstantDim)) if den_list else 1
        if den != 0 and num % den == 0:
            return ConstantDim(num // den)
        return None
    return ExpressionDim(
        f"({shape_product_repr(tuple(num_list))})/({shape_product_repr(tuple(den_list))})"
    )


def shape_product_repr(dims: tuple[Dim, ...]) -> str:
    if not dims:
        return "1"
    return "*".join(render_dim(dim) for dim in dims)


def sum_dim(dims: tuple[Dim, ...]) -> Dim:
    if all(isinstance(dim, ConstantDim) for dim in dims):
        return ConstantDim(sum(dim.value for dim in dims if isinstance(dim, ConstantDim)))
    return ExpressionDim("(" + " + ".join(render_dim(dim) for dim in dims) + ")")


def dims_compatible(left: Dim, right: Dim) -> bool:
    if isinstance(left, ConstantDim) and left.value == 1:
        return True
    if isinstance(right, ConstantDim) and right.value == 1:
        return True
    return render_dim(left) == render_dim(right)


def _broadcast_dim(left: Dim, right: Dim) -> Dim | None:
    """Return the output dim for a pair of broadcast-aligned dimensions.

    Returns None only for provably incompatible constant pairs.
    Symbolic vs. anything non-matching yields an UnknownDim (uncertain).
    """
    if isinstance(left, ConstantDim) and left.value == 1:
        return right
    if isinstance(right, ConstantDim) and right.value == 1:
        return left
    if render_dim(left) == render_dim(right):
        return left
    # Definite incompatibility: both constants, neither is 1.
    if isinstance(left, ConstantDim) and isinstance(right, ConstantDim):
        return None
    # Uncertain: at least one symbolic dim — prefer the constant if present.
    if isinstance(left, ConstantDim):
        return left
    if isinstance(right, ConstantDim):
        return right
    return UnknownDim("?")


def broadcast_has_uncertain_dims(left: TensorShape, right: TensorShape) -> bool:
    """Return True if any aligned dim pair is uncertain (symbolic vs. non-matching).

    Used by the analyzer to emit a warning instead of an error.
    """
    for ldim, rdim in zip(reversed(left.dims), reversed(right.dims), strict=False):
        if dims_compatible(ldim, rdim):
            continue
        if isinstance(ldim, ConstantDim) and isinstance(rdim, ConstantDim):
            continue  # definite mismatch — handled as error, not warning
        return True
    return False


def broadcast_shapes(left: TensorShape, right: TensorShape) -> TensorShape | None:
    """Compute the broadcast shape of two tensors following NumPy/PyTorch rules.

    Dimensions are aligned from the right; leading batch dims of the larger operand are
    prepended to the result. A size-1 dimension is compatible with any size.
    Symbolic dimensions that cannot be verified statically are kept as UnknownDim.

    Args:
        left: Shape of the left operand.
        right: Shape of the right operand.

    Returns:
        Broadcast TensorShape, or None only when a pair of constant dims is provably
        incompatible (both non-1 and unequal).
    """
    out: list[Dim] = []
    # zip stops at the shorter operand; unmatched leading dims are handled below
    for ldim, rdim in zip(reversed(left.dims), reversed(right.dims), strict=False):
        result = _broadcast_dim(ldim, rdim)
        if result is None:
            return None
        out.append(result)
    if left.rank > right.rank:
        out.extend(reversed(left.dims[: left.rank - right.rank]))
    elif right.rank > left.rank:
        out.extend(reversed(right.dims[: right.rank - left.rank]))
    out.reverse()
    return TensorShape(tuple(out))


def batch_matmul_shape(left: TensorShape, right: TensorShape) -> TensorShape | None:
    """Compute the output shape of a batched matrix multiplication.

    Args:
        left: Shape of the left operand. shape: (*, M, K)  (rank >= 2)
        right: Shape of the right operand. shape: (*, K, N)  (rank >= 2)

    Returns:
        Output TensorShape (*, M, N) where * is the broadcast of the batch prefixes,
        or None if ranks are too low, inner dims do not match, or batch dims are
        incompatible.
    """
    if left.rank < 2 or right.rank < 2:
        return None
    if render_dim(left.dims[-1]) != render_dim(right.dims[-2]):
        return None
    if left.rank == 2 and right.rank == 2:
        return TensorShape((left.dims[0], right.dims[1]))
    batch_left = TensorShape(left.dims[:-2])
    batch_right = TensorShape(right.dims[:-2])
    batch_dims = broadcast_shapes(batch_left, batch_right)
    if batch_dims is None:
        return None
    return TensorShape(batch_dims.dims + (left.dims[-2], right.dims[-1]))


def normalize_index(index: int, rank: int) -> int | None:
    """Convert a possibly-negative axis index to a non-negative one.

    Args:
        index: Axis index; negative values count from the end (-1 → last axis).
        rank: Total number of dimensions.

    Returns:
        Non-negative index in [0, rank), or None if the index is out of bounds.
    """
    adjusted = index if index >= 0 else rank + index
    if 0 <= adjusted < rank:
        return adjusted
    return None

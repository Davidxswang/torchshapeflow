from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TypeAlias


@dataclass(frozen=True)
class SymbolicDim:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class ConstantDim:
    value: int

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class UnknownDim:
    token: str

    def __str__(self) -> str:
        return self.token


@dataclass(frozen=True)
class ExpressionDim:
    expr: str

    def __str__(self) -> str:
        return self.expr


Dim: TypeAlias = SymbolicDim | ConstantDim | UnknownDim | ExpressionDim


@dataclass(frozen=True)
class TensorShape:
    dims: tuple[Dim, ...]

    @property
    def rank(self) -> int:
        return len(self.dims)

    def __str__(self) -> str:
        return "[" + ", ".join(render_dim(dim) for dim in self.dims) + "]"


@dataclass(frozen=True)
class TensorValue:
    shape: TensorShape
    # TODO: populate and use origin to provide richer hover diagnostics
    # (e.g. "inferred from param x")
    origin: str | None = None

    @property
    def rank(self) -> int:
        return self.shape.rank

    def describe(self) -> str:
        return f"Tensor{self.shape}"


@dataclass(frozen=True)
class ShapeTupleValue:
    dims: tuple[Dim, ...]


@dataclass(frozen=True)
class IntegerValue:
    value: int | None


@dataclass(frozen=True)
class LinearSpec:
    in_features: int
    out_features: int


@dataclass(frozen=True)
class Conv2dSpec:
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]


Value: TypeAlias = TensorValue | ShapeTupleValue | IntegerValue | LinearSpec | Conv2dSpec


def render_dim(dim: Dim) -> str:
    return str(dim)


def make_dim(value: str | int) -> Dim:
    if isinstance(value, int):
        return ConstantDim(value)
    return SymbolicDim(value)


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


def quotient_dim(numerator: tuple[Dim, ...], denominator: tuple[Dim, ...]) -> Dim:
    """Compute the quotient of two dimension-products.

    Used to infer the -1 dimension in a reshape. Returns a ConstantDim when both
    sequences are fully constant and the division is exact; otherwise returns an
    ExpressionDim with the formula as a string.

    Args:
        numerator: Dimensions whose product forms the numerator.
        denominator: Dimensions whose product forms the denominator.

    Returns:
        ConstantDim if the result is a whole number, ExpressionDim otherwise.
    """
    if all(isinstance(dim, ConstantDim) for dim in numerator + denominator):
        num = prod(dim.value for dim in numerator if isinstance(dim, ConstantDim))
        den = prod(dim.value for dim in denominator if isinstance(dim, ConstantDim))
        if den != 0 and num % den == 0:
            return ConstantDim(num // den)
    return ExpressionDim(f"({shape_product_repr(numerator)})/({shape_product_repr(denominator)})")


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


def broadcast_shapes(left: TensorShape, right: TensorShape) -> TensorShape | None:
    """Compute the broadcast shape of two tensors following NumPy/PyTorch rules.

    Dimensions are aligned from the right; leading batch dims of the larger operand are
    prepended to the result. A size-1 dimension is compatible with any size.

    Args:
        left: Shape of the left operand.
        right: Shape of the right operand.

    Returns:
        Broadcast TensorShape, or None if any aligned pair of dimensions is incompatible.
    """
    out: list[Dim] = []
    # zip stops at the shorter operand; unmatched leading dims are handled below
    for ldim, rdim in zip(reversed(left.dims), reversed(right.dims), strict=False):
        if dims_compatible(ldim, rdim):
            if isinstance(ldim, ConstantDim) and ldim.value == 1:
                out.append(rdim)
            else:
                out.append(ldim)
            continue
        return None
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

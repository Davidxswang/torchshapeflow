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
    # in_features is None when the constructor arg was non-literal (e.g. from a variable).
    in_features: int | None
    # out_features is str (symbolic label) when the constructor arg was a variable name.
    out_features: int | str


@dataclass(frozen=True)
class Conv2dSpec:
    # in_channels is None when the constructor arg was non-literal (e.g. from a config).
    in_channels: int | None
    # out_channels is str (symbolic label) when the constructor arg was a variable name.
    out_channels: int | str
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]


@dataclass(frozen=True)
class PassthroughSpec:
    """Marker for shape-preserving nn.Module instances (BatchNorm, LayerNorm, ReLU, etc.).

    The output shape equals the input shape; no validation is performed.
    """


@dataclass(frozen=True)
class EmbeddingSpec:
    """Spec for nn.Embedding — appends embedding_dim to the index tensor shape."""

    # embedding_dim is str (symbolic label) when the constructor arg was a variable name.
    embedding_dim: int | str


@dataclass(frozen=True)
class Pool2dSpec:
    """Spec for nn.MaxPool2d and nn.AvgPool2d — preserves (N, C), transforms (H, W).

    stride defaults to kernel_size in PyTorch when not explicitly provided.
    dilation is always (1, 1) for nn.AvgPool2d (that layer has no dilation parameter).
    """

    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]


@dataclass(frozen=True)
class MultiheadAttentionSpec:
    """Spec for nn.MultiheadAttention.

    When called with ``(query, key, value)``, output is a ``TensorTupleValue``
    where the first element has the same shape as ``query`` and the second element
    is the attention weights (shape unknown statically).
    """

    embed_dim: int
    num_heads: int
    batch_first: bool


@dataclass(frozen=True)
class LSTMSpec:
    """Spec for nn.LSTM.

    input_size is None when the constructor arg was non-literal.
    D = 2 if bidirectional else 1.
    H_out = proj_size when proj_size > 0, else hidden_size.
    Returns a nested TupleValue matching PyTorch:
      [0] output:     (N, L, D*H_out) if batch_first else (L, N, D*H_out)
      [1][0] h_n:     (D*num_layers, N, H_out)
      [1][1] c_n:     (D*num_layers, N, hidden_size)
    """

    input_size: int | None
    # hidden_size / proj_size / num_layers are str (symbolic labels) when the constructor arg
    # was a variable name.
    hidden_size: int | str
    proj_size: int | str | None
    num_layers: int | str
    batch_first: bool
    bidirectional: bool


@dataclass(frozen=True)
class CustomModuleSpec:
    """Spec derived from an annotated custom nn.Module.forward contract."""

    input_shape: TensorValue | None
    return_shape: TensorValue | None


@dataclass(frozen=True)
class RepeatSpec:
    """Spec for a module repeated a known or unknown number of times.

    When ``count`` is ``None`` or symbolic, inference may still succeed if applying
    ``spec`` once leaves the output shape unchanged thereafter (a shape fixed point).
    ``min_count`` captures lower bounds known statically, e.g. from ``if depth <= 0:
    raise`` before ``range(depth)``.
    """

    spec: ModuleSpec
    count: int | str | None
    min_count: int = 0


@dataclass(frozen=True)
class SequentialSpec:
    """Spec for nn.Sequential — applies a chain of sub-module specs in order."""

    specs: tuple[
        LinearSpec
        | Conv2dSpec
        | PassthroughSpec
        | EmbeddingSpec
        | Pool2dSpec
        | MultiheadAttentionSpec
        | LSTMSpec
        | CustomModuleSpec
        | RepeatSpec
        | SequentialSpec,
        ...,
    ]


# Convenience alias for any module spec type.
ModuleSpec: TypeAlias = (
    LinearSpec
    | Conv2dSpec
    | PassthroughSpec
    | EmbeddingSpec
    | Pool2dSpec
    | MultiheadAttentionSpec
    | LSTMSpec
    | CustomModuleSpec
    | RepeatSpec
    | SequentialSpec
)


@dataclass(frozen=True)
class TensorTupleValue:
    """A statically-known fixed-length tuple of tensors, e.g. from x.chunk() or x.split()."""

    tensors: tuple[TensorValue, ...]


@dataclass(frozen=True)
class TupleValue:
    """A statically-known fixed-length tuple of values, potentially nested."""

    items: tuple[Value, ...]


Value: TypeAlias = (
    TensorValue
    | ShapeTupleValue
    | IntegerValue
    | LinearSpec
    | Conv2dSpec
    | PassthroughSpec
    | EmbeddingSpec
    | Pool2dSpec
    | MultiheadAttentionSpec
    | LSTMSpec
    | CustomModuleSpec
    | RepeatSpec
    | SequentialSpec
    | TensorTupleValue
    | TupleValue
    | Dim
)


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

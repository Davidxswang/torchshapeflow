from __future__ import annotations

from dataclasses import dataclass
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

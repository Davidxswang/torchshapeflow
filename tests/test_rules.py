from __future__ import annotations

import ast

from torchshapeflow.model import (
    ConstantDim,
    Conv2dSpec,
    ExpressionDim,
    LinearSpec,
    ShapeTupleValue,
    SymbolicDim,
    TensorShape,
    TensorValue,
)
from torchshapeflow.rules.conv2d import infer_conv2d
from torchshapeflow.rules.indexing import infer_subscript
from torchshapeflow.rules.linear import infer_linear


def _t(*dims: int | str) -> TensorValue:
    """Build a TensorValue from int (constant) or str (symbolic) dimension specs."""
    shape = TensorShape(
        tuple(ConstantDim(d) if isinstance(d, int) else SymbolicDim(d) for d in dims)
    )
    return TensorValue(shape)


def _subscript_node(src: str) -> ast.Subscript:
    node = ast.parse(src, mode="eval").body
    assert isinstance(node, ast.Subscript)
    return node


# ---------------------------------------------------------------------------
# infer_linear
# ---------------------------------------------------------------------------


def test_linear_basic() -> None:
    spec = LinearSpec(in_features=768, out_features=256)
    result = infer_linear(spec, _t("B", "T", 768))
    assert result is not None
    assert str(result.shape) == "[B, T, 256]"


def test_linear_1d_input() -> None:
    spec = LinearSpec(in_features=4, out_features=8)
    result = infer_linear(spec, _t(4))
    assert result is not None
    assert str(result.shape) == "[8]"


def test_linear_wrong_in_features() -> None:
    spec = LinearSpec(in_features=768, out_features=256)
    assert infer_linear(spec, _t("B", "T", 512)) is None


def test_linear_rank_zero() -> None:
    spec = LinearSpec(in_features=4, out_features=8)
    t = TensorValue(TensorShape(()))
    assert infer_linear(spec, t) is None


def test_linear_symbolic_last_dim_rejected() -> None:
    # Symbolic last dim cannot be matched against a constant in_features
    spec = LinearSpec(in_features=768, out_features=256)
    assert infer_linear(spec, _t("B", "D")) is None


# ---------------------------------------------------------------------------
# infer_conv2d
# ---------------------------------------------------------------------------


def _conv_spec(
    in_channels: int = 3,
    out_channels: int = 8,
    kernel_size: tuple[int, int] = (3, 3),
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> Conv2dSpec:
    return Conv2dSpec(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def test_conv2d_with_padding_preserves_spatial() -> None:
    spec = _conv_spec(padding=(1, 1))
    result = infer_conv2d(spec, _t("B", 3, 32, 32))
    assert result is not None
    assert str(result.shape) == "[B, 8, 32, 32]"


def test_conv2d_without_padding_shrinks_spatial() -> None:
    result = infer_conv2d(_conv_spec(), _t("B", 3, 32, 32))
    assert result is not None
    assert str(result.shape) == "[B, 8, 30, 30]"


def test_conv2d_stride_2() -> None:
    spec = _conv_spec(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    result = infer_conv2d(spec, _t("B", 3, 32, 32))
    assert result is not None
    assert str(result.shape) == "[B, 8, 16, 16]"


def test_conv2d_wrong_channels() -> None:
    assert infer_conv2d(_conv_spec(), _t("B", 4, 32, 32)) is None


def test_conv2d_wrong_rank() -> None:
    assert infer_conv2d(_conv_spec(), _t("B", 3, 32)) is None


def test_conv2d_symbolic_spatial_dims() -> None:
    result = infer_conv2d(_conv_spec(), _t("B", 3, "H", "W"))
    assert result is not None
    # Spatial dims become expression strings
    assert isinstance(result.shape.dims[2], ExpressionDim)
    assert isinstance(result.shape.dims[3], ExpressionDim)


# ---------------------------------------------------------------------------
# infer_subscript — tensor indexing
# ---------------------------------------------------------------------------


def test_subscript_integer_removes_dim() -> None:
    result = infer_subscript(_t("B", "C", "H", "W"), _subscript_node("x[0]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[C, H, W]"


def test_subscript_slice_keeps_dim() -> None:
    result = infer_subscript(_t("B", "C"), _subscript_node("x[1:3]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[B, C]"


def test_subscript_none_inserts_dim() -> None:
    result = infer_subscript(_t("B", "C"), _subscript_node("x[None]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[1, B, C]"


def test_subscript_ellipsis_passthrough() -> None:
    result = infer_subscript(_t("B", "C", "H", "W"), _subscript_node("x[...]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[B, C, H, W]"


def test_subscript_mixed() -> None:
    # x[0, :, None] on shape [B, C, H]:
    #   0    -> removes B (position 0)
    #   :    -> keeps C   (position 1)
    #   None -> inserts 1 at current output position (before remaining H)
    # remaining dims [H] appended → [C, 1, H]
    result = infer_subscript(_t("B", "C", "H"), _subscript_node("x[0, :, None]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[C, 1, H]"


# ---------------------------------------------------------------------------
# infer_subscript — shape tuple indexing
# ---------------------------------------------------------------------------


def test_subscript_shape_tuple_positive() -> None:
    shape = ShapeTupleValue((ConstantDim(3), ConstantDim(4)))
    result = infer_subscript(shape, _subscript_node("x[1]"))
    assert result == ConstantDim(4)


def test_subscript_shape_tuple_negative() -> None:
    shape = ShapeTupleValue((ConstantDim(3), ConstantDim(4)))
    result = infer_subscript(shape, _subscript_node("x[-1]"))
    assert result == ConstantDim(4)


def test_subscript_shape_tuple_out_of_range() -> None:
    shape = ShapeTupleValue((ConstantDim(3),))
    result = infer_subscript(shape, _subscript_node("x[5]"))
    assert result is None

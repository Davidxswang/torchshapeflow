from __future__ import annotations

import ast

from torchshapeflow.model import (
    ConstantDim,
    Conv2dSpec,
    EmbeddingSpec,
    ExpressionDim,
    LinearSpec,
    LSTMSpec,
    Pool2dSpec,
    ShapeTupleValue,
    SymbolicDim,
    TensorShape,
    TensorValue,
    TupleValue,
)
from torchshapeflow.rules.conv2d import infer_conv2d
from torchshapeflow.rules.embedding import infer_embedding
from torchshapeflow.rules.indexing import infer_subscript
from torchshapeflow.rules.linear import infer_linear
from torchshapeflow.rules.lstm import infer_lstm
from torchshapeflow.rules.pool2d import infer_pool2d
from torchshapeflow.rules.shape_ops import infer_chunk, infer_reduction, infer_split


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


def test_linear_symbolic_last_dim_passes_through() -> None:
    # Symbolic last dim: in_features check is skipped; out_features is still propagated.
    spec = LinearSpec(in_features=768, out_features=256)
    result = infer_linear(spec, _t("B", "D"))
    assert result is not None
    assert str(result.shape) == "[B, 256]"


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


def test_subscript_slice_constant_computes_size() -> None:
    # x[1:3] on [B, C]: constant-bound slice computes size (3-1=2), rest kept.
    result = infer_subscript(_t("B", "C"), _subscript_node("x[1:3]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[2, C]"


def test_subscript_slice_no_upper_symbolic_keeps_dim() -> None:
    # x[1:] on [B, C]: first dim is symbolic, so cannot compute — preserved.
    result = infer_subscript(_t("B", "C"), _subscript_node("x[1:]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[B, C]"


def test_subscript_slice_no_upper_constant_computes() -> None:
    # x[1:] on [32, C]: first dim is constant 32, so 32-1=31.
    result = infer_subscript(_t(32, "C"), _subscript_node("x[1:]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[31, C]"


def test_subscript_slice_negative_lower_constant() -> None:
    # x[-2:] on [10, C]: 10 - (10-2) = 2.
    result = infer_subscript(_t(10, "C"), _subscript_node("x[-2:]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[2, C]"


def test_subscript_slice_negative_upper_constant() -> None:
    # x[:-1] on [10, C]: upper resolves to 10-1=9, lower is 0 → 9.
    result = infer_subscript(_t(10, "C"), _subscript_node("x[:-1]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[9, C]"


def test_subscript_slice_negative_both_constant() -> None:
    # x[-3:-1] on [10, C]: lower=10-3=7, upper=10-1=9 → 2.
    result = infer_subscript(_t(10, "C"), _subscript_node("x[-3:-1]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[2, C]"


def test_subscript_slice_negative_lower_out_of_range_clamps() -> None:
    # x[-100:] on [10, C]: -100 clamps to 0, so 10-0=10.
    result = infer_subscript(_t(10, "C"), _subscript_node("x[-100:]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[10, C]"


def test_subscript_slice_negative_upper_out_of_range_clamps() -> None:
    # x[:-100] on [10, C]: -100 clamps to 0, upper=0 not > lower=0 → preserved.
    result = infer_subscript(_t(10, "C"), _subscript_node("x[:-100]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[10, C]"


def test_subscript_slice_positive_upper_clamps_to_dim() -> None:
    # x[:100] on [10, C]: upper clamps to 10, lower is 0 → 10.
    result = infer_subscript(_t(10, "C"), _subscript_node("x[:100]"))
    assert result is not None
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[10, C]"


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


# ---------------------------------------------------------------------------
# infer_chunk
# ---------------------------------------------------------------------------


def test_chunk_divisible() -> None:
    result = infer_chunk(_t("B", "T", 192), n=3, dim=-1)
    assert result is not None
    assert len(result.tensors) == 3
    assert all(str(tv.shape) == "[B, T, 64]" for tv in result.tensors)


def test_chunk_symbolic_dim_expression() -> None:
    result = infer_chunk(_t("B", "T", "D"), n=3, dim=-1)
    assert result is not None
    assert len(result.tensors) == 3
    # Symbolic dim produces expression, rank preserved
    assert all(tv.rank == 3 for tv in result.tensors)
    assert all(str(tv.shape) == "[B, T, D//3]" for tv in result.tensors)


def test_chunk_non_divisible() -> None:
    result = infer_chunk(_t("B", 10), n=3, dim=-1)
    assert result is not None
    assert len(result.tensors) == 3
    # ceil(10/3) = 4 for first two chunks, remainder 2 for last
    assert str(result.tensors[0].shape) == "[B, 4]"
    assert str(result.tensors[1].shape) == "[B, 4]"
    assert str(result.tensors[2].shape) == "[B, 2]"


def test_chunk_out_of_range_returns_none() -> None:
    assert infer_chunk(_t("B", "T"), n=2, dim=5) is None


# ---------------------------------------------------------------------------
# infer_split
# ---------------------------------------------------------------------------


def test_split_int_size_divisible() -> None:
    result = infer_split(_t("B", "T", 192), split_size=64, dim=-1)
    assert result is not None
    assert len(result.tensors) == 3
    assert all(str(tv.shape) == "[B, T, 64]" for tv in result.tensors)


def test_split_int_size_non_divisible() -> None:
    result = infer_split(_t("B", 10), split_size=3, dim=-1)
    assert result is not None
    assert len(result.tensors) == 4  # 3+3+3+1


def test_split_list_sizes() -> None:
    result = infer_split(_t("B", "T", 10), split_size=[3, 7], dim=-1)
    assert result is not None
    assert len(result.tensors) == 2
    assert str(result.tensors[0].shape) == "[B, T, 3]"
    assert str(result.tensors[1].shape) == "[B, T, 7]"


def test_split_list_sum_mismatch_returns_none() -> None:
    # sections sum to 12 but dim is 10 — should be rejected
    assert infer_split(_t("B", "T", 10), split_size=[5, 7], dim=-1) is None


def test_split_symbolic_dim_returns_none() -> None:
    # Can't determine chunk count without knowing dim size
    assert infer_split(_t("B", "T", "D"), split_size=64, dim=-1) is None


def test_split_list_symbolic_ok() -> None:
    # list form works even with symbolic dims
    result = infer_split(_t("B", "T", "D"), split_size=[32, 32], dim=-1)
    assert result is not None
    assert len(result.tensors) == 2


# ---------------------------------------------------------------------------
# infer_reduction
# ---------------------------------------------------------------------------


def test_reduction_no_dim_scalar() -> None:
    result = infer_reduction(_t("B", "T", 64))
    assert result is not None
    assert result.rank == 0


def test_reduction_single_dim() -> None:
    result = infer_reduction(_t("B", "T", 64), dim=1)
    assert result is not None
    assert str(result.shape) == "[B, 64]"


def test_reduction_keepdim() -> None:
    result = infer_reduction(_t("B", "T", 64), dim=1, keepdim=True)
    assert result is not None
    assert str(result.shape) == "[B, 1, 64]"


def test_reduction_tuple_dim() -> None:
    result = infer_reduction(_t("B", 4, 4), dim=(1, 2))
    assert result is not None
    assert str(result.shape) == "[B]"


def test_reduction_negative_dim() -> None:
    result = infer_reduction(_t("B", "T", 64), dim=-1)
    assert result is not None
    assert str(result.shape) == "[B, T]"


def test_reduction_out_of_range() -> None:
    assert infer_reduction(_t("B", "T"), dim=5) is None


# ---------------------------------------------------------------------------
# infer_embedding
# ---------------------------------------------------------------------------


def test_embedding_appends_dim() -> None:
    spec = EmbeddingSpec(embedding_dim=64)
    result = infer_embedding(spec, _t("B", "T"))
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[B, T, 64]"


def test_embedding_scalar_index() -> None:
    spec = EmbeddingSpec(embedding_dim=128)
    result = infer_embedding(spec, _t("B"))
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[B, 128]"


# ---------------------------------------------------------------------------
# infer_pool2d
# ---------------------------------------------------------------------------


def test_pool2d_basic() -> None:
    spec = Pool2dSpec(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1))
    result = infer_pool2d(spec, _t("B", "C", 8, 8))
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[B, C, 4, 4]"


def test_pool2d_preserves_batch_channels() -> None:
    spec = Pool2dSpec(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1))
    result = infer_pool2d(spec, _t("B", "C", 16, 16))
    assert isinstance(result, TensorValue)
    assert str(result.shape) == "[B, C, 16, 16]"


def test_pool2d_rejects_non_4d() -> None:
    spec = Pool2dSpec(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1))
    assert infer_pool2d(spec, _t("B", "C", 8)) is None


# ---------------------------------------------------------------------------
# infer_lstm
# ---------------------------------------------------------------------------


def _lstm(
    hidden_size: int = 256,
    num_layers: int = 1,
    batch_first: bool = False,
    bidirectional: bool = False,
    input_size: int | None = 128,
    proj_size: int | None = None,
) -> LSTMSpec:
    return LSTMSpec(
        input_size=input_size,
        hidden_size=hidden_size,
        proj_size=proj_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
    )


def test_lstm_returns_three_tensors() -> None:
    result = infer_lstm(_lstm(), _t("L", "N", 128))
    assert isinstance(result, TupleValue)
    assert len(result.items) == 2


def test_lstm_seq_first_shapes() -> None:
    result = infer_lstm(_lstm(hidden_size=256), _t("L", "N", 128))
    assert result is not None
    output = result.items[0]
    state = result.items[1]
    assert isinstance(output, TensorValue)
    assert isinstance(state, TupleValue)
    h_n, c_n = state.items
    assert isinstance(h_n, TensorValue)
    assert isinstance(c_n, TensorValue)
    assert str(output.shape) == "[L, N, 256]"
    assert str(h_n.shape) == "[1, N, 256]"
    assert str(c_n.shape) == "[1, N, 256]"


def test_lstm_batch_first_shapes() -> None:
    result = infer_lstm(_lstm(hidden_size=256, batch_first=True), _t("N", "L", 128))
    assert result is not None
    output = result.items[0]
    state = result.items[1]
    assert isinstance(output, TensorValue)
    assert isinstance(state, TupleValue)
    h_n = state.items[0]
    assert isinstance(h_n, TensorValue)
    assert str(output.shape) == "[N, L, 256]"
    assert str(h_n.shape) == "[1, N, 256]"


def test_lstm_bidirectional_doubles_output_hidden() -> None:
    result = infer_lstm(_lstm(hidden_size=256, bidirectional=True), _t("L", "N", 128))
    assert result is not None
    output = result.items[0]
    state = result.items[1]
    assert isinstance(output, TensorValue)
    assert isinstance(state, TupleValue)
    h_n = state.items[0]
    assert isinstance(h_n, TensorValue)
    assert output.shape.dims[-1] == ConstantDim(512)  # D*hidden = 2*256
    assert h_n.shape.dims[0] == ConstantDim(2)  # D*num_layers = 2*1


def test_lstm_num_layers_affects_h_n() -> None:
    result = infer_lstm(_lstm(num_layers=3), _t("L", "N", 128))
    assert result is not None
    state = result.items[1]
    assert isinstance(state, TupleValue)
    h_n, c_n = state.items
    assert isinstance(h_n, TensorValue)
    assert isinstance(c_n, TensorValue)
    assert h_n.shape.dims[0] == ConstantDim(3)
    assert c_n.shape.dims[0] == ConstantDim(3)


def test_lstm_bidirectional_multilayer() -> None:
    result = infer_lstm(_lstm(hidden_size=128, num_layers=2, bidirectional=True), _t("L", "N", 128))
    assert result is not None
    state = result.items[1]
    assert isinstance(state, TupleValue)
    h_n = state.items[0]
    assert isinstance(h_n, TensorValue)
    assert h_n.shape.dims[0] == ConstantDim(4)  # D*num_layers = 2*2


def test_lstm_rejects_non_rank3() -> None:
    assert infer_lstm(_lstm(), _t("N", "L")) is None
    assert infer_lstm(_lstm(), _t("N", "L", 128, 4)) is None


def test_lstm_input_size_none_skips_check() -> None:
    result = infer_lstm(_lstm(input_size=None), _t("L", "N", "D"))
    assert result is not None


def test_lstm_input_size_mismatch_rejects() -> None:
    assert infer_lstm(_lstm(input_size=128), _t("L", "N", 64)) is None


def test_lstm_proj_size_changes_output_and_hidden_but_not_cell() -> None:
    result = infer_lstm(_lstm(hidden_size=256, proj_size=64), _t("L", "N", 128))
    assert result is not None
    output = result.items[0]
    state = result.items[1]
    assert isinstance(output, TensorValue)
    assert isinstance(state, TupleValue)
    h_n, c_n = state.items
    assert isinstance(h_n, TensorValue)
    assert isinstance(c_n, TensorValue)
    assert str(output.shape) == "[L, N, 64]"
    assert str(h_n.shape) == "[1, N, 64]"
    assert str(c_n.shape) == "[1, N, 256]"

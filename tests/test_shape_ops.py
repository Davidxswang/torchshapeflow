from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    IntegerValue,
    ShapeTupleValue,
    SymbolicDim,
    TensorShape,
    TensorValue,
    UnknownDim,
)
from torchshapeflow.rules.shape_ops import (
    infer_cat,
    infer_diagonal,
    infer_einsum,
    infer_flatten,
    infer_index_select,
    infer_interpolate,
    infer_matmul,
    infer_mm,
    infer_movedim,
    infer_one_hot,
    infer_permute,
    infer_reduction,
    infer_reshape,
    infer_size,
    infer_squeeze,
    infer_stack,
    infer_topk,
    infer_transpose,
    infer_unsqueeze,
)


def _t(*dims: int | str) -> TensorValue:
    """Build a TensorValue from int (constant) or str (symbolic) dimension specs."""
    shape = TensorShape(
        tuple(ConstantDim(d) if isinstance(d, int) else SymbolicDim(d) for d in dims)
    )
    return TensorValue(shape)


# ---------------------------------------------------------------------------
# infer_permute
# ---------------------------------------------------------------------------


def test_permute_basic() -> None:
    result = infer_permute(_t("B", "C", "H", "W"), (0, 2, 3, 1))
    assert result is not None
    assert str(result.shape) == "[B, H, W, C]"


def test_permute_negative_indices() -> None:
    result = infer_permute(_t("B", "C", "H"), (0, -1, 1))
    assert result is not None
    assert str(result.shape) == "[B, H, C]"


def test_permute_wrong_length() -> None:
    assert infer_permute(_t("B", "C", "H"), (0, 1)) is None


def test_permute_duplicate_axes() -> None:
    assert infer_permute(_t("B", "C", "H"), (0, 0, 1)) is None


def test_permute_out_of_range() -> None:
    assert infer_permute(_t("B", "C"), (0, 5)) is None


# ---------------------------------------------------------------------------
# infer_transpose
# ---------------------------------------------------------------------------


def test_transpose_basic() -> None:
    result = infer_transpose(_t("B", "H", "T", "D"), -2, -1)
    assert result is not None
    assert str(result.shape) == "[B, H, D, T]"


def test_transpose_positive_indices() -> None:
    result = infer_transpose(_t("B", "C", "H", "W"), 1, 2)
    assert result is not None
    assert str(result.shape) == "[B, H, C, W]"


def test_transpose_out_of_range() -> None:
    assert infer_transpose(_t("B", "C"), 0, 5) is None


# ---------------------------------------------------------------------------
# infer_reshape
# ---------------------------------------------------------------------------


def test_reshape_constant_valid() -> None:
    result = infer_reshape(_t(2, 3, 4), (6, 4))
    assert result is not None
    assert str(result.shape) == "[6, 4]"


def test_reshape_constant_product_mismatch() -> None:
    # 2*3*4=24, 3*9=27 — must be rejected
    assert infer_reshape(_t(2, 3, 4), (3, 9)) is None


def test_reshape_minus_one_infers_dim() -> None:
    result = infer_reshape(_t(2, 3, 4), (-1, 4))
    assert result is not None
    assert str(result.shape) == "[6, 4]"


def test_reshape_two_minus_one_rejected() -> None:
    assert infer_reshape(_t(2, 3, 4), (-1, -1)) is None


def test_reshape_symbolic_skips_constant_check() -> None:
    # Input has a symbolic dim — product cannot be checked statically; must not error
    result = infer_reshape(_t("B", 3, 4), (SymbolicDim("B"), -1))
    assert result is not None


def test_reshape_all_symbolic_no_minus_one() -> None:
    result = infer_reshape(_t("B", "C"), (SymbolicDim("B"), SymbolicDim("C")))
    assert result is not None
    assert str(result.shape) == "[B, C]"


def test_reshape_symbolic_cancel_minus_one() -> None:
    # [B, 3, 4] → [B, 6, -1]: B cancels, 12/6 = 2.
    result = infer_reshape(_t("B", 3, 4), (SymbolicDim("B"), 6, -1))
    assert result is not None
    assert str(result.shape) == "[B, 6, 2]"


def test_reshape_symbolic_cancel_multiple() -> None:
    # [B, C, 3, 4] → [B, C, -1]: B,C cancel, 12/1 = 12.
    result = infer_reshape(_t("B", "C", 3, 4), (SymbolicDim("B"), SymbolicDim("C"), -1))
    assert result is not None
    assert str(result.shape) == "[B, C, 12]"


def test_reshape_symbolic_cancel_invalid() -> None:
    # [B, 3, 4] → [B, 5, -1]: B cancels, 12/5 not divisible → None.
    result = infer_reshape(_t("B", 3, 4), (SymbolicDim("B"), 5, -1))
    assert result is None


# ---------------------------------------------------------------------------
# infer_flatten
# ---------------------------------------------------------------------------


def test_flatten_full() -> None:
    result = infer_flatten(_t(2, 3, 4))
    assert result is not None
    assert str(result.shape) == "[24]"


def test_flatten_partial_constant() -> None:
    result = infer_flatten(_t("B", 3, 4), 1, -1)
    assert result is not None
    assert str(result.shape) == "[B, 12]"


def test_flatten_start_after_end() -> None:
    assert infer_flatten(_t("B", 3, 4), 2, 1) is None


def test_flatten_out_of_range() -> None:
    assert infer_flatten(_t("B", 3), 0, 5) is None


# ---------------------------------------------------------------------------
# infer_squeeze
# ---------------------------------------------------------------------------


def test_squeeze_no_dim_removes_all_ones() -> None:
    result = infer_squeeze(_t(1, 3, 1, 4))
    assert result is not None
    assert str(result.shape) == "[3, 4]"


def test_squeeze_specific_dim() -> None:
    result = infer_squeeze(_t(1, 3, 4), 0)
    assert result is not None
    assert str(result.shape) == "[3, 4]"


def test_squeeze_negative_index() -> None:
    result = infer_squeeze(_t(3, 1), -1)
    assert result is not None
    assert str(result.shape) == "[3]"


def test_squeeze_non_one_dim_rejected() -> None:
    assert infer_squeeze(_t(2, 3, 4), 0) is None


def test_squeeze_out_of_range() -> None:
    assert infer_squeeze(_t(1, 3), 5) is None


def test_squeeze_symbolic_dim_returns_tensor_unchanged() -> None:
    # Can't verify a symbolic dim is 1 — return input unchanged rather than drop
    result = infer_squeeze(_t("B", "C", "H"), 0)
    assert result is not None
    assert str(result.shape) == "[B, C, H]"


# ---------------------------------------------------------------------------
# infer_unsqueeze
# ---------------------------------------------------------------------------


def test_unsqueeze_front() -> None:
    result = infer_unsqueeze(_t(3, 4), 0)
    assert result is not None
    assert str(result.shape) == "[1, 3, 4]"


def test_unsqueeze_middle() -> None:
    result = infer_unsqueeze(_t(3, 4), 1)
    assert result is not None
    assert str(result.shape) == "[3, 1, 4]"


def test_unsqueeze_end_positive() -> None:
    result = infer_unsqueeze(_t(3, 4), 2)
    assert result is not None
    assert str(result.shape) == "[3, 4, 1]"


def test_unsqueeze_end_negative() -> None:
    # -1 should insert at the end for a rank-2 tensor
    result = infer_unsqueeze(_t(3, 4), -1)
    assert result is not None
    assert str(result.shape) == "[3, 4, 1]"


def test_unsqueeze_second_to_last_negative() -> None:
    result = infer_unsqueeze(_t(3, 4), -2)
    assert result is not None
    assert str(result.shape) == "[3, 1, 4]"


def test_unsqueeze_out_of_range() -> None:
    assert infer_unsqueeze(_t(3, 4), 5) is None


# ---------------------------------------------------------------------------
# infer_size
# ---------------------------------------------------------------------------


def test_size_no_dim() -> None:
    result = infer_size(_t(2, 3))
    assert isinstance(result, ShapeTupleValue)
    assert result.dims == (ConstantDim(2), ConstantDim(3))


def test_size_constant_dim() -> None:
    result = infer_size(_t(2, 3), 1)
    assert result == IntegerValue(3)


def test_size_negative_index() -> None:
    result = infer_size(_t(2, 3), -1)
    assert result == IntegerValue(3)


def test_size_symbolic_dim_returns_integer_value_none() -> None:
    # Symbolic dim: return IntegerValue(None) so the variable stays tracked
    result = infer_size(_t("B", 3), 0)
    assert result == IntegerValue(None)


def test_size_out_of_range() -> None:
    assert infer_size(_t(2, 3), 5) is None


# ---------------------------------------------------------------------------
# infer_cat
# ---------------------------------------------------------------------------


def test_cat_basic() -> None:
    # Both concat dims are constant; sum_dim reduces to ConstantDim(8)
    result = infer_cat((_t("B", 3, 4), _t("B", 5, 4)), 1)
    assert result is not None
    assert str(result.shape) == "[B, 8, 4]"


def test_cat_single_tensor() -> None:
    # sum_dim over one symbolic dim wraps it: SymbolicDim("B") -> ExpressionDim("(B)")
    result = infer_cat((_t("B", 3),), 0)
    assert result is not None
    assert str(result.shape) == "[(B), 3]"


def test_cat_empty_returns_none() -> None:
    assert infer_cat((), 0) is None


def test_cat_rank_mismatch() -> None:
    assert infer_cat((_t("B", 3), _t("B", 3, 4)), 0) is None


def test_cat_dim_mismatch_on_non_cat_axis() -> None:
    assert infer_cat((_t("B", 3, 4), _t("B", 5, 5)), 1) is None


def test_cat_negative_dim() -> None:
    # Both concat dims are constant; sum_dim reduces to ConstantDim(10)
    result = infer_cat((_t("B", 3, 4), _t("B", 3, 6)), -1)
    assert result is not None
    assert str(result.shape) == "[B, 3, 10]"


# ---------------------------------------------------------------------------
# infer_stack
# ---------------------------------------------------------------------------


def test_stack_at_dim_zero() -> None:
    result = infer_stack((_t("B", 3), _t("B", 3)), 0)
    assert result is not None
    assert str(result.shape) == "[2, B, 3]"


def test_stack_at_end_negative() -> None:
    result = infer_stack((_t("B", 3), _t("B", 3)), -1)
    assert result is not None
    assert str(result.shape) == "[B, 3, 2]"


def test_stack_shape_mismatch() -> None:
    assert infer_stack((_t("B", 3), _t("B", 4)), 0) is None


def test_stack_empty() -> None:
    assert infer_stack((), 0) is None


def test_stack_out_of_range() -> None:
    assert infer_stack((_t("B", 3), _t("B", 3)), 5) is None


# ---------------------------------------------------------------------------
# infer_matmul
# ---------------------------------------------------------------------------


def test_matmul_2d() -> None:
    result = infer_matmul(_t(3, 4), _t(4, 5))
    assert result is not None
    assert str(result.shape) == "[3, 5]"


def test_matmul_batched() -> None:
    result = infer_matmul(_t("B", 3, 4), _t("B", 4, 5))
    assert result is not None
    assert str(result.shape) == "[B, 3, 5]"


def test_matmul_inner_dim_mismatch() -> None:
    assert infer_matmul(_t(3, 4), _t(5, 6)) is None


def test_matmul_rank_too_low() -> None:
    assert infer_matmul(_t(3), _t(3)) is None


# ---------------------------------------------------------------------------
# infer_reduction
# ---------------------------------------------------------------------------


def test_reduction_no_dim_scalar() -> None:
    result = infer_reduction(_t("B", 4, 4))
    assert result is not None
    assert str(result.shape) == "[]"


def test_reduction_no_dim_keepdim() -> None:
    result = infer_reduction(_t("B", 4, 4), keepdim=True)
    assert result is not None
    assert str(result.shape) == "[1, 1, 1]"


def test_reduction_single_dim() -> None:
    result = infer_reduction(_t("B", "T", 768), dim=1)
    assert result is not None
    assert str(result.shape) == "[B, 768]"


def test_reduction_negative_dim() -> None:
    result = infer_reduction(_t("B", "T", 768), dim=-1)
    assert result is not None
    assert str(result.shape) == "[B, T]"


def test_reduction_keepdim() -> None:
    result = infer_reduction(_t("B", "T", 768), dim=1, keepdim=True)
    assert result is not None
    assert str(result.shape) == "[B, 1, 768]"


def test_reduction_tuple_dim() -> None:
    result = infer_reduction(_t("B", 4, 4), dim=(1, 2))
    assert result is not None
    assert str(result.shape) == "[B]"


def test_reduction_tuple_dim_keepdim() -> None:
    result = infer_reduction(_t("B", 4, 4), dim=(1, 2), keepdim=True)
    assert result is not None
    assert str(result.shape) == "[B, 1, 1]"


def test_reduction_symbolic_dim_preserved() -> None:
    result = infer_reduction(_t("B", "H", "W"), dim=0)
    assert result is not None
    assert str(result.shape) == "[H, W]"


def test_reduction_out_of_range() -> None:
    assert infer_reduction(_t("B", 4), dim=5) is None


def test_reduction_duplicate_dims() -> None:
    assert infer_reduction(_t("B", 4, 4), dim=(1, 1)) is None


# ---------------------------------------------------------------------------
# infer_mm
# ---------------------------------------------------------------------------


def test_mm_basic() -> None:
    result = infer_mm(_t(3, 4), _t(4, 5))
    assert result is not None
    assert str(result.shape) == "[3, 5]"


def test_mm_symbolic() -> None:
    result = infer_mm(_t("M", "K"), _t("K", "N"))
    assert result is not None
    assert str(result.shape) == "[M, N]"


def test_mm_inner_dim_mismatch() -> None:
    assert infer_mm(_t(3, 4), _t(5, 6)) is None


def test_mm_requires_rank2() -> None:
    assert infer_mm(_t(2, 3, 4), _t(4, 5)) is None
    assert infer_mm(_t(3, 4), _t(4)) is None


# ---------------------------------------------------------------------------
# infer_movedim
# ---------------------------------------------------------------------------


def test_movedim_single() -> None:
    # move axis 1 to position 0: (B, C, H, W) → (C, B, H, W)
    result = infer_movedim(_t("B", "C", "H", "W"), 1, 0)
    assert result is not None
    assert str(result.shape) == "[C, B, H, W]"


def test_movedim_identity() -> None:
    result = infer_movedim(_t("B", "T", "D"), 1, 1)
    assert result is not None
    assert str(result.shape) == "[B, T, D]"


def test_movedim_negative() -> None:
    # move last axis (-1) to front (0): (B, T, D) → (D, B, T)
    result = infer_movedim(_t("B", "T", "D"), -1, 0)
    assert result is not None
    assert str(result.shape) == "[D, B, T]"


def test_movedim_tuple() -> None:
    # move axes (0, 1) to (1, 2): (A, B, C) → (C, A, B)
    result = infer_movedim(_t("A", "B", "C"), (0, 1), (1, 2))
    assert result is not None
    assert str(result.shape) == "[C, A, B]"


def test_movedim_out_of_bounds() -> None:
    assert infer_movedim(_t("B", "T"), 5, 0) is None


def test_movedim_length_mismatch() -> None:
    assert infer_movedim(_t("B", "T", "D"), (0, 1), (2,)) is None


# ---------------------------------------------------------------------------
# infer_einsum
# ---------------------------------------------------------------------------


def test_einsum_bmm() -> None:
    # "bik,bkj->bij": batched matrix multiply (B, T, D) @ (B, D, T) -> (B, T, T)
    q = _t("B", "T", "D")
    k = _t("B", "D", "T")
    result = infer_einsum("bik,bkj->bij", [q, k])
    assert result is not None
    assert str(result.shape) == "[B, T, T]"


def test_einsum_matrix_vector() -> None:
    # "ij,j->i"
    A = _t(4, 8)
    v = _t(8)
    result = infer_einsum("ij,j->i", [A, v])
    assert result is not None
    assert str(result.shape) == "[4]"


def test_einsum_outer_product() -> None:
    # "i,j->ij"
    result = infer_einsum("i,j->ij", [_t("M"), _t("N")])
    assert result is not None
    assert str(result.shape) == "[M, N]"


def test_einsum_trace() -> None:
    # "ii->" (scalar output)
    result = infer_einsum("ii->", [_t(4, 4)])
    assert result is not None
    assert str(result.shape) == "[]"


def test_einsum_contraction_dim_mismatch() -> None:
    # K=8 vs K=9 — should return None
    q = _t("B", "T", 8)
    k = _t("B", 9, "T")
    assert infer_einsum("bik,bkj->bij", [q, k]) is None


def test_einsum_wrong_operand_count() -> None:
    assert infer_einsum("ij,jk->ik", [_t(2, 3)]) is None


def test_einsum_implicit_mode_unsupported() -> None:
    # No "->" → not supported yet
    assert infer_einsum("ij,jk", [_t(2, 3), _t(3, 4)]) is None


def test_einsum_label_count_mismatch() -> None:
    # "ijk" has 3 labels but tensor has rank 2
    assert infer_einsum("ijk,jk->ik", [_t(2, 3), _t(3, 4)]) is None


# ---------------------------------------------------------------------------
# infer_interpolate
# ---------------------------------------------------------------------------


def test_interpolate_constant_size() -> None:
    result = infer_interpolate(
        _t("B", "C", 32, 32), size=(ConstantDim(64), ConstantDim(64)), scale_factor=None
    )
    assert result is not None and str(result.shape) == "[B, C, 64, 64]"


def test_interpolate_scale_factor() -> None:
    result = infer_interpolate(_t("B", "C", 32, 32), size=None, scale_factor=(2.0, 2.0))
    assert result is not None and str(result.shape) == "[B, C, 64, 64]"


def test_interpolate_symbolic_spatial() -> None:
    result = infer_interpolate(_t("B", "C", "H", "W"), size=None, scale_factor=(0.5, 0.5))
    assert result is not None and str(result.shape) == "[B, C, ?, ?]"


def test_interpolate_requires_rank3() -> None:
    assert infer_interpolate(_t("B", "C"), size=(ConstantDim(32),), scale_factor=None) is None


# ---------------------------------------------------------------------------
# infer_one_hot
# ---------------------------------------------------------------------------


def test_one_hot_basic() -> None:
    result = infer_one_hot(_t("B", "H", "W"), num_classes=64)
    assert result is not None and str(result.shape) == "[B, H, W, 64]"


# ---------------------------------------------------------------------------
# infer_diagonal
# ---------------------------------------------------------------------------


def test_diagonal_square() -> None:
    result = infer_diagonal(_t(4, 4), offset=0, dim1=0, dim2=1)
    assert result is not None and str(result.shape) == "[4]"


def test_diagonal_rectangular() -> None:
    result = infer_diagonal(_t(3, 5), offset=0, dim1=0, dim2=1)
    assert result is not None and str(result.shape) == "[3]"


def test_diagonal_with_batch() -> None:
    # (..., H, W) diagonal on last two dims
    result = infer_diagonal(_t("B", 4, 4), offset=0, dim1=-2, dim2=-1)
    assert result is not None and str(result.shape) == "[B, 4]"


def test_diagonal_offset() -> None:
    result = infer_diagonal(_t(4, 4), offset=1, dim1=0, dim2=1)
    assert result is not None and str(result.shape) == "[3]"


# ---------------------------------------------------------------------------
# infer_index_select
# ---------------------------------------------------------------------------


def test_index_select_constant() -> None:
    result = infer_index_select(_t("B", 64, "H"), dim=1, index_len=ConstantDim(10))
    assert result is not None and str(result.shape) == "[B, 10, H]"


def test_index_select_unknown() -> None:
    result = infer_index_select(_t("B", 64), dim=1, index_len=UnknownDim("?"))
    assert result is not None and str(result.shape) == "[B, ?]"


# ---------------------------------------------------------------------------
# infer_topk
# ---------------------------------------------------------------------------


def test_topk_last_dim() -> None:
    result = infer_topk(_t("B", 64), k=10, dim=-1)
    assert result is not None and str(result.shape) == "[B, 10]"


def test_topk_explicit_dim() -> None:
    result = infer_topk(_t("B", "T", 64), k=5, dim=2)
    assert result is not None and str(result.shape) == "[B, T, 5]"

from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    IntegerValue,
    ShapeTupleValue,
    SymbolicDim,
    TensorShape,
    TensorValue,
)
from torchshapeflow.rules.shape_ops import (
    infer_cat,
    infer_flatten,
    infer_matmul,
    infer_permute,
    infer_reshape,
    infer_size,
    infer_squeeze,
    infer_stack,
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


def test_size_symbolic_dim_returns_none() -> None:
    assert infer_size(_t("B", 3), 0) is None


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

from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    SymbolicDim,
    TensorShape,
    TensorValue,
)
from torchshapeflow.rules.concat import infer_cat, infer_stack


def _t(*dims: int | str) -> TensorValue:
    """Build a TensorValue from int (constant) or str (symbolic) dimension specs."""
    shape = TensorShape(
        tuple(ConstantDim(d) if isinstance(d, int) else SymbolicDim(d) for d in dims)
    )
    return TensorValue(shape)


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

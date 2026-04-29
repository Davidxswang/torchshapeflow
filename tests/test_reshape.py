from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    IntegerValue,
    ShapeTupleValue,
    SymbolicDim,
    TensorShape,
    TensorValue,
)
from torchshapeflow.rules.reshape import (
    infer_flatten,
    infer_reshape,
    infer_size,
    infer_squeeze,
    infer_unsqueeze,
)


def _t(*dims: int | str) -> TensorValue:
    """Build a TensorValue from int (constant) or str (symbolic) dimension specs."""
    shape = TensorShape(
        tuple(ConstantDim(d) if isinstance(d, int) else SymbolicDim(d) for d in dims)
    )
    return TensorValue(shape)


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

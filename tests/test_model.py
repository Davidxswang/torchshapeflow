from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    ExpressionDim,
    SymbolicDim,
    TensorShape,
    batch_matmul_shape,
    broadcast_shapes,
    normalize_index,
    product_dim,
    quotient_dim,
    sum_dim,
)


def test_broadcast_shapes() -> None:
    left = TensorShape((SymbolicDim("B"), ConstantDim(1), ConstantDim(4)))
    right = TensorShape((SymbolicDim("B"), ConstantDim(3), ConstantDim(4)))
    broadcast = broadcast_shapes(left, right)
    assert broadcast is not None
    assert str(broadcast) == "[B, 3, 4]"


def test_broadcast_shapes_rank_mismatch() -> None:
    left = TensorShape((ConstantDim(3), ConstantDim(4)))
    right = TensorShape((ConstantDim(1), ConstantDim(3), ConstantDim(4)))
    broadcast = broadcast_shapes(left, right)
    assert broadcast is not None
    assert str(broadcast) == "[1, 3, 4]"


def test_broadcast_shapes_incompatible() -> None:
    left = TensorShape((ConstantDim(3), ConstantDim(4)))
    right = TensorShape((ConstantDim(5), ConstantDim(4)))
    assert broadcast_shapes(left, right) is None


# ---------------------------------------------------------------------------
# product_dim
# ---------------------------------------------------------------------------


def test_product_dim_all_constants() -> None:
    assert product_dim((ConstantDim(2), ConstantDim(3), ConstantDim(4))) == ConstantDim(24)


def test_product_dim_single_constant() -> None:
    assert product_dim((ConstantDim(5),)) == ConstantDim(5)


def test_product_dim_symbolic() -> None:
    result = product_dim((SymbolicDim("B"), SymbolicDim("C")))
    assert str(result) == "B*C"


def test_product_dim_constant_factor_comes_first() -> None:
    # Constant factor must appear first in the output string
    result = product_dim((SymbolicDim("B"), ConstantDim(4)))
    assert str(result) == "4*B"


def test_product_dim_constant_factor_1_omitted() -> None:
    # A constant factor of 1 should not appear in the output
    result = product_dim((SymbolicDim("B"), ConstantDim(1)))
    assert str(result) == "B"


# ---------------------------------------------------------------------------
# quotient_dim
# ---------------------------------------------------------------------------


def test_quotient_dim_exact_division() -> None:
    assert quotient_dim((ConstantDim(12),), (ConstantDim(4),)) == ConstantDim(3)


def test_quotient_dim_non_divisible_is_none() -> None:
    result = quotient_dim((ConstantDim(10),), (ConstantDim(3),))
    assert result is None


def test_quotient_dim_symbolic_is_expression() -> None:
    result = quotient_dim((SymbolicDim("B"), ConstantDim(4)), (ConstantDim(2),))
    assert isinstance(result, ExpressionDim)


def test_quotient_dim_symbolic_cancel_to_constant() -> None:
    # (B * 3 * 4) / (B * 6) → B cancels, 12 / 6 = 2
    result = quotient_dim(
        (SymbolicDim("B"), ConstantDim(3), ConstantDim(4)),
        (SymbolicDim("B"), ConstantDim(6)),
    )
    assert result == ConstantDim(2)


def test_quotient_dim_symbolic_cancel_multiple() -> None:
    # (B * C * 12) / (B * C * 4) → B,C cancel, 12 / 4 = 3
    result = quotient_dim(
        (SymbolicDim("B"), SymbolicDim("C"), ConstantDim(12)),
        (SymbolicDim("B"), SymbolicDim("C"), ConstantDim(4)),
    )
    assert result == ConstantDim(3)


def test_quotient_dim_symbolic_cancel_invalid() -> None:
    # (B * 3 * 4) / (B * 5) → B cancels, 12 / 5 not divisible → None
    result = quotient_dim(
        (SymbolicDim("B"), ConstantDim(3), ConstantDim(4)),
        (SymbolicDim("B"), ConstantDim(5)),
    )
    assert result is None


# ---------------------------------------------------------------------------
# sum_dim
# ---------------------------------------------------------------------------


def test_sum_dim_constants() -> None:
    assert sum_dim((ConstantDim(3), ConstantDim(4))) == ConstantDim(7)


def test_sum_dim_symbolic() -> None:
    result = sum_dim((SymbolicDim("A"), SymbolicDim("B")))
    assert str(result) == "(A + B)"


# ---------------------------------------------------------------------------
# normalize_index
# ---------------------------------------------------------------------------


def test_normalize_index_positive() -> None:
    assert normalize_index(0, 4) == 0
    assert normalize_index(3, 4) == 3


def test_normalize_index_negative() -> None:
    assert normalize_index(-1, 4) == 3
    assert normalize_index(-4, 4) == 0


def test_normalize_index_out_of_bounds() -> None:
    assert normalize_index(4, 4) is None
    assert normalize_index(-5, 4) is None


# ---------------------------------------------------------------------------
# batch_matmul_shape
# ---------------------------------------------------------------------------


def test_batch_matmul_2d() -> None:
    left = TensorShape((ConstantDim(3), ConstantDim(4)))
    right = TensorShape((ConstantDim(4), ConstantDim(5)))
    result = batch_matmul_shape(left, right)
    assert result is not None
    assert str(result) == "[3, 5]"


def test_batch_matmul_batched() -> None:
    left = TensorShape((SymbolicDim("B"), ConstantDim(3), ConstantDim(4)))
    right = TensorShape((SymbolicDim("B"), ConstantDim(4), ConstantDim(5)))
    result = batch_matmul_shape(left, right)
    assert result is not None
    assert str(result) == "[B, 3, 5]"


def test_batch_matmul_inner_dim_mismatch() -> None:
    left = TensorShape((ConstantDim(3), ConstantDim(4)))
    right = TensorShape((ConstantDim(5), ConstantDim(6)))
    assert batch_matmul_shape(left, right) is None


def test_batch_matmul_rank_too_low() -> None:
    left = TensorShape((ConstantDim(3),))
    right = TensorShape((ConstantDim(3),))
    assert batch_matmul_shape(left, right) is None

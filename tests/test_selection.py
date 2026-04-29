from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    SymbolicDim,
    TensorShape,
    TensorValue,
    UnknownDim,
)
from torchshapeflow.rules.selection import (
    infer_diagonal,
    infer_index_select,
    infer_interpolate,
    infer_one_hot,
    infer_topk,
)


def _t(*dims: int | str) -> TensorValue:
    """Build a TensorValue from int (constant) or str (symbolic) dimension specs."""
    shape = TensorShape(
        tuple(ConstantDim(d) if isinstance(d, int) else SymbolicDim(d) for d in dims)
    )
    return TensorValue(shape)


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


def test_diagonal_symbolic_same_dim_no_offset() -> None:
    # Both dims are the same symbolic dim and offset=0 → diagonal length equals that dim
    result = infer_diagonal(_t("B", "D", "D"), offset=0, dim1=-2, dim2=-1)
    assert result is not None and str(result.shape) == "[B, D]"


def test_diagonal_symbolic_different_dims() -> None:
    # Different symbolic dims → unknown
    result = infer_diagonal(_t("B", "H", "W"), offset=0, dim1=-2, dim2=-1)
    assert result is not None and str(result.shape) == "[B, ?]"


def test_diagonal_symbolic_with_offset() -> None:
    # Same symbolic dim but non-zero offset → unknown (can't compute min-|offset|)
    result = infer_diagonal(_t("B", "D", "D"), offset=1, dim1=-2, dim2=-1)
    assert result is not None and str(result.shape) == "[B, ?]"


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


# ---------------------------------------------------------------------------
# infer_one_hot
# ---------------------------------------------------------------------------


def test_one_hot_basic() -> None:
    result = infer_one_hot(_t("B", "H", "W"), num_classes=64)
    assert result is not None and str(result.shape) == "[B, H, W, 64]"


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


def test_interpolate_symbolic_spatial_integer_scale() -> None:
    # Integer scale factor with symbolic spatial → expression
    result = infer_interpolate(_t("B", "C", "H", "W"), size=None, scale_factor=(2.0, 2.0))
    assert result is not None and str(result.shape) == "[B, C, 2*H, 2*W]"


def test_interpolate_symbolic_spatial_non_integer_scale() -> None:
    # Non-integer scale factor with symbolic spatial → unknown
    result = infer_interpolate(_t("B", "C", "H", "W"), size=None, scale_factor=(0.5, 0.5))
    assert result is not None and str(result.shape) == "[B, C, ?, ?]"


def test_interpolate_requires_rank3() -> None:
    assert infer_interpolate(_t("B", "C"), size=(ConstantDim(32),), scale_factor=None) is None

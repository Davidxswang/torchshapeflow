from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    SymbolicDim,
    TensorShape,
    TensorValue,
)
from torchshapeflow.rules.transpose import infer_movedim, infer_permute, infer_transpose


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

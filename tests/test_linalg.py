from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    SymbolicDim,
    TensorShape,
    TensorValue,
)
from torchshapeflow.rules.linalg import (
    infer_einsum,
    infer_matmul,
    infer_mm,
    infer_reduction,
)


def _t(*dims: int | str) -> TensorValue:
    """Build a TensorValue from int (constant) or str (symbolic) dimension specs."""
    shape = TensorShape(
        tuple(ConstantDim(d) if isinstance(d, int) else SymbolicDim(d) for d in dims)
    )
    return TensorValue(shape)


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


def test_einsum_symbolic_dim_conflict() -> None:
    # Same label maps to different symbolic dims → rejected
    assert infer_einsum("ij,jk->ik", [_t("M", "K"), _t("L", "N")]) is None


def test_einsum_symbolic_dim_match() -> None:
    # Same label maps to same symbolic dim → accepted
    result = infer_einsum("ij,jk->ik", [_t("M", "K"), _t("K", "N")])
    assert result is not None
    assert str(result.shape) == "[M, N]"


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

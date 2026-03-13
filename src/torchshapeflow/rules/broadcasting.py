from __future__ import annotations

from torchshapeflow.model import TensorValue, broadcast_shapes


def infer_binary_broadcast(left: TensorValue, right: TensorValue) -> TensorValue | None:
    """Infer shape: left.shape, right.shape -> broadcast(left, right)."""
    shape = broadcast_shapes(left.shape, right.shape)
    if shape is None:
        return None
    return TensorValue(shape)

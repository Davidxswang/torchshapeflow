from __future__ import annotations

from torchshapeflow.arithmetic import broadcast_shapes
from torchshapeflow.model import TensorValue


def infer_binary_broadcast(left: TensorValue, right: TensorValue) -> TensorValue | None:
    """Infer shape: left.shape, right.shape -> broadcast(left, right)."""
    shape = broadcast_shapes(left.shape, right.shape)
    if shape is None:
        return None
    return TensorValue(shape)

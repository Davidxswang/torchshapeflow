from __future__ import annotations

from torchshapeflow.model import ConstantDim, LinearSpec, TensorShape, TensorValue


def infer_linear(spec: LinearSpec, tensor: TensorValue) -> TensorValue | None:
    """Infer shape: (..., in_features) -> (..., out_features)."""
    if tensor.rank == 0:
        return None
    if tensor.shape.dims[-1] != ConstantDim(spec.in_features):
        return None
    dims = list(tensor.shape.dims[:-1])
    dims.append(ConstantDim(spec.out_features))
    return TensorValue(TensorShape(tuple(dims)))

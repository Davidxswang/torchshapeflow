from __future__ import annotations

from torchshapeflow.model import ConstantDim, LinearSpec, TensorShape, TensorValue


def infer_linear(spec: LinearSpec, tensor: TensorValue) -> TensorValue | None:
    """Infer shape: (..., in_features) -> (..., out_features).

    The in_features check is skipped when spec.in_features is None (non-literal
    constructor arg) or when the input's last dim is not a ConstantDim. In both
    cases inference still runs and out_features is propagated.
    """
    if tensor.rank == 0:
        return None
    last_dim = tensor.shape.dims[-1]
    if spec.in_features is not None and isinstance(last_dim, ConstantDim):
        if last_dim != ConstantDim(spec.in_features):
            return None
    dims = list(tensor.shape.dims[:-1])
    dims.append(ConstantDim(spec.out_features))
    return TensorValue(TensorShape(tuple(dims)))

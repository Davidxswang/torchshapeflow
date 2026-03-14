from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    Conv2dSpec,
    TensorShape,
    TensorValue,
)
from torchshapeflow.rules.common import spatial_output_dim


def infer_conv2d(spec: Conv2dSpec, tensor: TensorValue) -> TensorValue | None:
    """Infer shape: (N, C_in, H, W) -> (N, C_out, H_out, W_out).

    The in_channels check is skipped when spec.in_channels is None (non-literal
    constructor arg) or when the input's channel dim is not a ConstantDim. In both
    cases inference still runs and out_channels plus spatial dims are propagated.
    """
    if tensor.rank != 4:
        return None
    channels = tensor.shape.dims[1]
    if spec.in_channels is not None and isinstance(channels, ConstantDim):
        if channels != ConstantDim(spec.in_channels):
            return None
    height = spatial_output_dim(
        tensor.shape.dims[2],
        spec.kernel_size[0],
        spec.stride[0],
        spec.padding[0],
        spec.dilation[0],
    )
    width = spatial_output_dim(
        tensor.shape.dims[3],
        spec.kernel_size[1],
        spec.stride[1],
        spec.padding[1],
        spec.dilation[1],
    )
    return TensorValue(
        TensorShape((tensor.shape.dims[0], ConstantDim(spec.out_channels), height, width))
    )

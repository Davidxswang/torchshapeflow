from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    Conv2dSpec,
    Dim,
    ExpressionDim,
    TensorShape,
    TensorValue,
)


def infer_conv2d(spec: Conv2dSpec, tensor: TensorValue) -> TensorValue | None:
    """Infer shape: (N, C_in, H, W) -> (N, C_out, H_out, W_out)."""
    if tensor.rank != 4:
        return None
    channels = tensor.shape.dims[1]
    if channels != ConstantDim(spec.in_channels):
        return None
    height = _conv_dim(
        tensor.shape.dims[2],
        spec.kernel_size[0],
        spec.stride[0],
        spec.padding[0],
        spec.dilation[0],
    )
    width = _conv_dim(
        tensor.shape.dims[3],
        spec.kernel_size[1],
        spec.stride[1],
        spec.padding[1],
        spec.dilation[1],
    )
    return TensorValue(
        TensorShape((tensor.shape.dims[0], ConstantDim(spec.out_channels), height, width))
    )


def _conv_dim(dim: Dim, kernel: int, stride: int, padding: int, dilation: int) -> Dim:
    """Apply the convolution output-size formula to a single spatial dimension.

    Formula: floor((dim + 2*padding - dilation*(kernel-1) - 1) / stride + 1)

    Args:
        dim: Input spatial dimension.
        kernel: Kernel size.
        stride: Stride.
        padding: Zero-padding on each side.
        dilation: Dilation factor.

    Returns:
        ConstantDim with the computed integer size if dim is constant;
        ExpressionDim with the formula string otherwise.
    """
    if isinstance(dim, ConstantDim):
        value = ((dim.value + (2 * padding) - (dilation * (kernel - 1)) - 1) // stride) + 1
        return ConstantDim(value)
    return ExpressionDim(f"floor(({dim} + 2*{padding} - {dilation}*({kernel}-1) - 1)/{stride} + 1)")

from __future__ import annotations

from torchshapeflow.model import Pool2dSpec, TensorShape, TensorValue
from torchshapeflow.rules.common import spatial_output_dim


def infer_pool2d(spec: Pool2dSpec, tensor: TensorValue) -> TensorValue | None:
    """Infer shape after a 2-D pooling operation (MaxPool2d or AvgPool2d).

    The channel dimension is preserved unchanged; the spatial dimensions are
    transformed using the same formula as convolution.

    Formula: H_out = floor((H + 2*padding - dilation*(kernel-1) - 1) / stride + 1)

    Args:
        spec: Pooling spec with kernel_size, stride, padding, and dilation.
        tensor: Input tensor. shape: (N, C, H, W)

    Returns:
        Output tensor with N and C unchanged and H, W transformed.
        shape: (N, C, H_out, W_out)
        None if the tensor is not rank 4.
    """
    if tensor.rank != 4:
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
    return TensorValue(TensorShape((tensor.shape.dims[0], tensor.shape.dims[1], height, width)))

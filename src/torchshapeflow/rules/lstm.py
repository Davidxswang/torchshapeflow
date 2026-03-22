from __future__ import annotations

from torchshapeflow.model import (
    ConstantDim,
    LSTMSpec,
    TensorShape,
    TensorValue,
    TupleValue,
)
from torchshapeflow.rules.common import scale_dim, to_dim


def infer_lstm(spec: LSTMSpec, tensor: TensorValue) -> TupleValue | None:
    """Infer shapes for nn.LSTM output.

    Args:
        spec: LSTM spec with hidden_size, num_layers, batch_first, bidirectional.
        tensor: Input tensor.
            shape: (L, N, input_size) if batch_first=False
            shape: (N, L, input_size) if batch_first=True

    Returns:
        TupleValue matching PyTorch's nested return structure:
          [0] output       — shape: (L, N, D*H_out) or (N, L, D*H_out) if batch_first
          [1][0] h_n       — shape: (D*num_layers, N, H_out)
          [1][1] c_n       — shape: (D*num_layers, N, hidden_size)
        None if the tensor is not rank 3 or the trailing input dimension definitely
        mismatches input_size.
        D = 2 if bidirectional else 1.
    """
    if tensor.rank != 3:
        return None

    d = 2 if spec.bidirectional else 1
    n_dim = tensor.shape.dims[0] if spec.batch_first else tensor.shape.dims[1]
    l_dim = tensor.shape.dims[1] if spec.batch_first else tensor.shape.dims[0]
    input_dim = tensor.shape.dims[-1]
    if (
        spec.input_size is not None
        and isinstance(input_dim, ConstantDim)
        and input_dim != to_dim(spec.input_size)
    ):
        return None

    state_size = spec.proj_size if spec.proj_size is not None else spec.hidden_size
    d_hidden = scale_dim(d, state_size)
    hidden = to_dim(state_size)
    cell_hidden = to_dim(spec.hidden_size)
    d_layers = scale_dim(d, spec.num_layers)

    if spec.batch_first:
        output = TensorValue(TensorShape((n_dim, l_dim, d_hidden)))
    else:
        output = TensorValue(TensorShape((l_dim, n_dim, d_hidden)))

    h_n = TensorValue(TensorShape((d_layers, n_dim, hidden)))
    c_n = TensorValue(TensorShape((d_layers, n_dim, cell_hidden)))

    return TupleValue((output, TupleValue((h_n, c_n))))

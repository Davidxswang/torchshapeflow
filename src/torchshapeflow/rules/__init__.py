from torchshapeflow.rules.broadcasting import infer_binary_broadcast
from torchshapeflow.rules.concat import infer_cat, infer_chunk, infer_split, infer_stack
from torchshapeflow.rules.conv2d import infer_conv2d
from torchshapeflow.rules.embedding import infer_embedding
from torchshapeflow.rules.indexing import infer_subscript
from torchshapeflow.rules.linalg import infer_einsum, infer_matmul, infer_mm, infer_reduction
from torchshapeflow.rules.linear import infer_linear
from torchshapeflow.rules.lstm import infer_lstm
from torchshapeflow.rules.pool2d import infer_pool2d
from torchshapeflow.rules.reshape import (
    infer_flatten,
    infer_reshape,
    infer_size,
    infer_squeeze,
    infer_unsqueeze,
)
from torchshapeflow.rules.selection import (
    infer_diagonal,
    infer_index_select,
    infer_interpolate,
    infer_one_hot,
    infer_topk,
)
from torchshapeflow.rules.transpose import infer_movedim, infer_permute, infer_transpose

__all__ = [
    "infer_binary_broadcast",
    "infer_cat",
    "infer_chunk",
    "infer_conv2d",
    "infer_diagonal",
    "infer_einsum",
    "infer_embedding",
    "infer_flatten",
    "infer_index_select",
    "infer_interpolate",
    "infer_linear",
    "infer_lstm",
    "infer_matmul",
    "infer_mm",
    "infer_movedim",
    "infer_one_hot",
    "infer_permute",
    "infer_pool2d",
    "infer_reduction",
    "infer_reshape",
    "infer_size",
    "infer_split",
    "infer_squeeze",
    "infer_stack",
    "infer_subscript",
    "infer_topk",
    "infer_transpose",
    "infer_unsqueeze",
]

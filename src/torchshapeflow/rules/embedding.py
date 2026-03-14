from __future__ import annotations

from torchshapeflow.model import ConstantDim, EmbeddingSpec, TensorShape, TensorValue


def infer_embedding(spec: EmbeddingSpec, tensor: TensorValue) -> TensorValue:
    """Infer shape after an embedding lookup.

    Args:
        spec: Embedding spec carrying the output embedding dimension.
        tensor: Index tensor of any rank. shape: (*indices)

    Returns:
        Output tensor with embedding_dim appended as a new trailing dimension.
        shape: (*indices, embedding_dim)
    """
    return TensorValue(TensorShape(tensor.shape.dims + (ConstantDim(spec.embedding_dim),)))

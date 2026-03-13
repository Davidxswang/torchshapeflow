from typing import Annotated

import torch

from torchshapeflow import Shape


def attention_projection(x: Annotated[torch.Tensor, Shape("B", "T", 768)]) -> torch.Tensor:
    q = x.reshape(x.shape[0], x.shape[1], 12, 64)
    k = q.permute(0, 2, 1, 3)
    return k

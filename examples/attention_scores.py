from typing import Annotated

import torch

from torchshapeflow import Shape


def attention_scores(
    q: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
    k: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
) -> torch.Tensor:
    scores = q.matmul(k.transpose(-2, -1))
    return scores

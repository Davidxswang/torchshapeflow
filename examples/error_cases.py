from typing import Annotated

import torch

from torchshapeflow import Shape


def bad_reshape(x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]) -> torch.Tensor:
    y = x.reshape(-1, -1)
    return y


def bad_broadcast(
    x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)],
    y: Annotated[torch.Tensor, Shape("B", 4, 32, 32)],
) -> torch.Tensor:
    return x + y

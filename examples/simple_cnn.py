from typing import Annotated

import torch
import torch.nn as nn

from torchshapeflow import Shape


class Net(nn.Module):
    def __init__(self) -> None:
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.linear = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]) -> torch.Tensor:
        y = self.conv(x)
        z = y.flatten(1)
        return self.linear(z)

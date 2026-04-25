"""Cleanly annotated CNN block — Conv2d + BatchNorm + ReLU residual."""

from __future__ import annotations

from typing import Annotated

import torch
from torch import nn

from torchshapeflow import Shape


class ConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(
        self,
        x: Annotated[torch.Tensor, Shape("B", "C", "H", "W")],
    ) -> Annotated[torch.Tensor, Shape("B", "C", "H", "W")]:
        h = self.bn1(self.conv1(x)).relu()
        h = self.bn2(self.conv2(h))
        return (x + h).relu()

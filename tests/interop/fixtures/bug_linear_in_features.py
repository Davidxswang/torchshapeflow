"""Deliberately-broken fixture: ``nn.Linear`` ``in_features`` mismatch.

TSF should report TSF1007 with structured ``expected`` / ``actual`` / ``hint``
fields. pyright / mypy / ty should each be silent — base type is still
``torch.Tensor``, and shape correctness is outside their domain.

Used by ``tests/interop/`` to verify (a) TSF catches the bug, (b) other
type-checkers do not produce false positives on the same source.
"""

from __future__ import annotations

from typing import Annotated

import torch
from torch import nn

from torchshapeflow import Shape


class Mismatch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(768, 256)

    def forward(
        self,
        x: Annotated[torch.Tensor, Shape("B", "T", 512)],
    ) -> Annotated[torch.Tensor, Shape("B", "T", 256)]:
        return self.fc(x)

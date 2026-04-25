"""Cleanly annotated transformer block — the canonical TSF use case.

Used by ``tests/interop/`` to verify that:
- pyright, mypy, ty all type-check this file with zero errors.
- TSF's analyzer reports zero diagnostics.

Kept tight to operations TSF tracks fully (Linear, matmul, transpose, addition
with consistent symbolic dims). Activations like ``.relu`` / ``.softmax`` are
TSF2001 inference gaps by design and would muddy a "clean" fixture; they live
in the real-world fixtures instead.
"""

from __future__ import annotations

from typing import Annotated

import torch
from torch import nn

from torchshapeflow import Shape


def attention_scores(
    q: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
    k: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
) -> Annotated[torch.Tensor, Shape("B", "H", "T", "T")]:
    return q @ k.transpose(-2, -1)


def attention_combine(
    weights: Annotated[torch.Tensor, Shape("B", "H", "T", "T")],
    v: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
) -> Annotated[torch.Tensor, Shape("B", "H", "T", "D")]:
    return weights @ v


class Projection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(768, 768)
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 768)

    def forward(
        self,
        x: Annotated[torch.Tensor, Shape("B", "T", 768)],
    ) -> Annotated[torch.Tensor, Shape("B", "T", 768)]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        combined = q + k + v
        return self.out_proj(combined)

"""Deliberately-broken fixture: matmul inner-dimension mismatch.

TSF should report TSF1003 with structured ``expected`` / ``actual`` / ``hint``
fields. pyright / mypy / ty should each be silent.
"""

from __future__ import annotations

from typing import Annotated

import torch

from torchshapeflow import Shape


def bad_attention(
    q: Annotated[torch.Tensor, Shape("B", "H", "T", 64)],
    k: Annotated[torch.Tensor, Shape("B", "H", 128, "T")],
) -> Annotated[torch.Tensor, Shape("B", "H", "T", "T")]:
    return q @ k

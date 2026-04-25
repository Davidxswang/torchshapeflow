"""Cleanly annotated module that exercises the canonical ``shapes.py`` recipe.

Demonstrates ``TypeAlias`` shape vocabularies — the recommended pattern for
larger projects per ``docs/syntax.md#type-alias-pattern``. Every aligned
dimension uses a literal size that matches the constructor literal so TSF
can verify symbolic / constant unification end-to-end.
"""

from __future__ import annotations

from typing import Annotated, TypeAlias

import torch
from torch import nn

from torchshapeflow import Shape

TokenSequence: TypeAlias = Annotated[torch.Tensor, Shape("B", "T")]
Embedding: TypeAlias = Annotated[torch.Tensor, Shape("B", "T", 768)]


class TokenEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(50_257, 768)

    def forward(self, tokens: TokenSequence) -> Embedding:
        return self.embed(tokens)


def project(
    embedded: Embedding,
    weight: Annotated[torch.Tensor, Shape(768, 256)],
) -> Annotated[torch.Tensor, Shape("B", "T", 256)]:
    return embedded @ weight

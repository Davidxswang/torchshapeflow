from typing import Annotated

import torch

from torchshapeflow import Shape


def patch_embed(x: Annotated[torch.Tensor, Shape("B", 3, 224, 224)]) -> torch.Tensor:
    patches = x.reshape(x.shape[0], 3, 14, 16, 14, 16)
    return patches.permute(0, 2, 4, 1, 3, 5)

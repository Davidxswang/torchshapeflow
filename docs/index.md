# TorchShapeFlow

TorchShapeFlow is a static, AST-based shape analyzer for PyTorch. It reads your Python source — no execution required — infers tensor shapes through your code, and reports mismatches as structured diagnostics.

```python
from typing import Annotated
import torch
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.linear = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]):
        y = self.conv(x)      # inferred: [B, 8, 32, 32]
        z = y.flatten(1)      # inferred: [B, 8192]
        return self.linear(z) # inferred: [B, 10]
```

```bash
$ tsf check mymodel.py
mymodel.py: ok

$ tsf check broken.py
broken.py:9:9 error TSF1004 Invalid reshape.
```

## What it does

- Reads `Annotated[torch.Tensor, Shape(...)]` annotations from function parameters
- Propagates symbolic shapes through [supported PyTorch operations](operators.md)
- Emits diagnostics when shapes are incompatible
- Provides hover-style shape facts for [editor integration](extension.md)

## Getting started

- [Quickstart](quickstart.md) — install and run your first check
- [Annotation syntax](syntax.md) — how to annotate your tensors
- [Supported operators](operators.md) — what is analyzed and what shapes are inferred

## For contributors

- [Architecture](architecture.md) — module map, analysis pipeline, Dim type system
- [Development](development.md) — make targets, CI, how to add a new operator

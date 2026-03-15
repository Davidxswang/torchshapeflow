# TorchShapeFlow

TorchShapeFlow is a static, AST-based shape analyzer for PyTorch. It reads your
Python source, infers tensor shapes from `Annotated[..., Shape(...)]`
contracts, and reports mismatches as structured diagnostics. No execution
required.

```python
from typing import Annotated
import torch
from torchshapeflow import Shape

def attention_scores(
    q: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
    k: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
) -> Annotated[torch.Tensor, Shape("B", "H", "T", "T")]:
    return q @ k.transpose(-2, -1)
```

```bash
$ tsf check mymodel.py
All clean (1 file checked)

$ tsf check broken.py
broken.py:9:9 error TSF1004 Invalid reshape.
```

## Philosophy

Like Pydantic for data validation, TorchShapeFlow is **annotation-first**:
you declare shape contracts on function parameters, and the analyzer verifies
consistency. Without annotations, there is nothing to check — and that is by
design. You opt in where it matters, starting with `forward`, and extend
coverage incrementally. Symbolic dimensions (`"B"`, `"T"`, `"D"`) are the
primary mechanism; the analyzer verifies that operations are consistent without
needing concrete sizes. Constants still matter, but mainly for semantically
fixed axes like channels, head counts, or embedding widths.

## What it does

- Reads `Annotated[torch.Tensor, Shape(...)]` contracts from function parameters
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

# Torch Shape Flow

Static PyTorch tensor shape diagnostics and hover information — without running your code.

## Requirements

The extension requires the `torchshapeflow` Python package to be installed:

```bash
pip install torchshapeflow
```

Or, if your project uses a virtual environment:

```bash
uv add torchshapeflow        # uv projects
pip install torchshapeflow   # pip/venv projects
```

## CLI discovery

The extension looks for `tsf` in this order:

1. `.venv/bin/tsf` in your workspace root (picked up automatically if you use a local virtual environment)
2. The path configured in `torchShapeFlow.cliPath` (see Settings below)
3. `tsf` on your system `PATH`

If your project uses a local `.venv`, no configuration is needed.

## Features

- **Diagnostics** — shape mismatches and invalid operations are highlighted inline as errors or warnings
- **Hover** — hover over tensor variables, annotated parameters, and function signatures to see inferred shapes

## Usage

The analyzer runs automatically when you save a Python file. You can also trigger it manually:

- Open the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`) and run **Torch Shape Flow: Run Analysis**

## Settings

| Setting | Default | Description |
|---|---|---|
| `torchShapeFlow.cliPath` | `tsf` | Path to the `tsf` executable, if not on PATH or in `.venv` |
| `torchShapeFlow.runOnSave` | `true` | Run the analyzer automatically on save |

## Annotation syntax

Annotate your function parameters with `Shape` to enable shape inference. In
practice, symbolic dimensions are the default path for model code:

```python
from typing import Annotated
import torch
from torchshapeflow import Shape

def attention_scores(
    q: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
    k: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
):
    scores = q @ k.transpose(-2, -1)  # hover shows: [B, H, T, T]
    return scores
```

See the [full documentation](https://davidxswang.github.io/torchshapeflow) for supported operators and annotation syntax.

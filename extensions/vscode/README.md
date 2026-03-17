# Torch Shape Flow

Static PyTorch tensor shape diagnostics and hover information — without running your code.

Torch Shape Flow is annotation-first and symbolic-first. You declare tensor
shape contracts with `Annotated[...]`, usually starting at model boundaries, and
the extension surfaces the inferred consequences inline as diagnostics and
hovers.

## Requirements

No separate Python package installation is required for normal use. The
published extension ships with bundled `tsf` executables for its supported
targets and uses them automatically.

Current bundled targets: Linux x64, macOS arm64, and Windows x64.
Other platforms can still use a workspace `.venv`, `cliPath`, or `tsf` on
`PATH`.

## CLI discovery

The extension looks for `tsf` in this order:

1. The path configured in `torchShapeFlow.cliPath`, if set
2. `.venv/bin/tsf` or `.venv/Scripts/tsf.exe` in your workspace root
3. The bundled executable shipped with the extension
4. `tsf` on your system `PATH`

This keeps the extension zero-install by default while still letting local
development environments override the executable. Bundled release binaries are
smoke-tested during CI and release builds before the `.vsix` is published.

## Features

- **Diagnostics** — shape mismatches and invalid operations are highlighted inline as errors or warnings
- **Hover** — hover over tensor variables, shape aliases, annotated parameters, and function signatures to see inferred shapes

## Usage

The analyzer runs automatically when you open or save a Python file. You can
also trigger it manually:

- Open the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`) and run **Torch Shape Flow: Run Analysis**

Annotate tensor boundaries with `Annotated[...]` to enable shape inference. In
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

The extension also shows hovers for shape aliases and annotated local variables
when those contracts are present in the active file.

## Settings

| Setting | Default | Description |
|---|---|---|
| `torchShapeFlow.cliPath` | `""` | Optional path to a specific `tsf` executable. When unset, the extension uses its normal discovery order. |
| `torchShapeFlow.runOnSave` | `true` | Run the analyzer automatically on save |

## Learn more

- [Full documentation](https://davidxswang.github.io/torchshapeflow)
- [Annotation syntax](https://davidxswang.github.io/torchshapeflow/syntax/)
- [Supported operators](https://davidxswang.github.io/torchshapeflow/operators/)
- [GitHub repository](https://github.com/Davidxswang/torchshapeflow)

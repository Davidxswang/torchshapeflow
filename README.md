# TorchShapeFlow

[![CI](https://github.com/Davidxswang/torchshapeflow/actions/workflows/ci.yml/badge.svg)](https://github.com/Davidxswang/torchshapeflow/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torchshapeflow)](https://pypi.org/project/torchshapeflow/)
[![Python](https://img.shields.io/pypi/pyversions/torchshapeflow)](https://pypi.org/project/torchshapeflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

TorchShapeFlow is a static, AST-based shape analyzer for PyTorch. It reads your Python source — no execution required — infers tensor shapes through your code, and reports mismatches as structured diagnostics.

```python
from typing import Annotated
import torch
import torch.nn as nn
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
```

## Install

```bash
pip install torchshapeflow
```

## Documentation

Full docs at **[davidxswang.github.io/torchshapeflow](https://davidxswang.github.io/torchshapeflow)**

- [Quickstart](https://davidxswang.github.io/torchshapeflow/quickstart/) — install and run your first check
- [Annotation syntax](https://davidxswang.github.io/torchshapeflow/syntax/) — how to annotate your tensors
- [Supported operators](https://davidxswang.github.io/torchshapeflow/operators/) — what is analyzed and what shapes are inferred
- [Limitations](https://davidxswang.github.io/torchshapeflow/limitations/) — what the analyzer does not handle

## Contributing

```bash
git clone https://github.com/Davidxswang/torchshapeflow
cd torchshapeflow
make install   # uv sync --extra dev
make check     # format + lint + typecheck + tests
```

See [docs/development.md](docs/development.md) for the full development guide: all make targets, CI workflow descriptions, and how to add new operators.

## Release

See [RELEASING.md](RELEASING.md) for the full release procedure.

Build commands:

- `make python-dist` — wheel and sdist into `dist/`
- `make extension-package` — VS Code extension `.vsix`
- `make build` — both

Marketplace publishing in the release workflow is gated on GitHub Actions secrets:

- `VSCE_PAT` for the VS Code Marketplace
- `OVSX_PAT` for Open VSX

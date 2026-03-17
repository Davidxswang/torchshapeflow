# TorchShapeFlow

[![CI](https://github.com/Davidxswang/torchshapeflow/actions/workflows/ci.yml/badge.svg)](https://github.com/Davidxswang/torchshapeflow/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/torchshapeflow?logo=pypi)](https://pypi.org/project/torchshapeflow/)
[![Python](https://img.shields.io/pypi/pyversions/torchshapeflow?logo=python)](https://pypi.org/project/torchshapeflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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
```

## Philosophy

TorchShapeFlow is annotation-first and symbolic-first.

- You declare tensor shape contracts with `Annotated[torch.Tensor, Shape(...)]`.
- Symbolic dimensions like `"B"`, `"T"`, and `"D"` are the default path for
  config-driven model code.
- Integer dimensions are still useful for fixed semantics like RGB channels or
  known embedding widths.
- When inference is not possible, the analyzer degrades visibly instead of
  guessing.

If Pydantic gives structure to data boundaries, TorchShapeFlow aims to do the
same for tensor-shape boundaries in deep learning code.

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

If you want to execute the example PyTorch scripts in `examples/`, install the
separate examples extra:

```bash
uv sync --extra dev --extra examples
```

See [docs/development.md](docs/development.md) for the full development guide: all make targets, CI workflow descriptions, and how to add new operators.

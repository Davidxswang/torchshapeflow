# Quickstart

## Install

```bash
pip install torchshapeflow
```

Or, from source:

```bash
git clone https://github.com/Davidxswang/torchshapeflow
cd torchshapeflow
make install   # uv sync --extra dev
```

## Annotate your tensors

TorchShapeFlow reads `Annotated[torch.Tensor, Shape(...)]` parameter annotations. String dimensions are symbolic; integer dimensions are constant:

```python
from typing import Annotated
import torch
from torchshapeflow import Shape

def attention(
    q: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
    k: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
) -> torch.Tensor:
    scores = q.matmul(k.transpose(-2, -1))  # [B, H, T, T]
    return scores
```

## Run the checker

```bash
tsf check path/to/mymodel.py
```

For machine-readable output (used by editor integrations):

```bash
tsf check path/to/mymodel.py --json
```

Check an entire directory:

```bash
tsf check src/
```

## Example output

**No errors:**

```
mymodel.py: ok
```

**With errors:**

```
broken.py:9:9 error TSF1004 Invalid reshape.
broken.py:17:12 error TSF1006 Broadcasting incompatibility.
```

Exit code is `0` when no errors are found, `1` otherwise.

## Try the bundled examples

```bash
tsf check examples/simple_cnn.py
tsf check examples/transformer_block.py
tsf check examples/error_cases.py --json
```

## Next steps

- [Annotation syntax](syntax.md) — all supported annotation forms
- [Supported operators](operators.md) — what shapes are tracked and how
- [Limitations](limitations.md) — what the analyzer does not handle

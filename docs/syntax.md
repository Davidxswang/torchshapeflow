# Annotation Syntax

## Basic form

Annotate function parameters using `typing.Annotated` with a `Shape(...)` metadata object:

```python
from typing import Annotated
import torch
from torchshapeflow import Shape

def forward(x: Annotated[torch.Tensor, Shape("B", "C", "H", "W")]) -> torch.Tensor:
    ...
```

`typing_extensions.Annotated` is also accepted for compatibility.

## Dimension types

| Form | Kind | Meaning |
|---|---|---|
| `"B"` | Symbolic | A named unknown size — batch, sequence length, etc. |
| `32` | Constant | A fixed integer size |

Symbolic names are case-sensitive: `"B"` and `"b"` are treated as different dimensions. There is no aliasing or unification — `"B"` in one function is independent of `"B"` in another.

## Mixed dimensions

String and integer dimensions can be freely combined:

```python
# (batch, channels=3, height, width)
Shape("B", 3, "H", "W")

# (batch, sequence, embedding=768)
Shape("B", "T", 768)
```

## Type alias pattern

For shapes used in multiple places, define a `TypeAlias`:

```python
from typing import Annotated, TypeAlias
import torch
from torchshapeflow import Shape

ImageBatch: TypeAlias = Annotated[torch.Tensor, Shape("B", 3, "H", "W")]
FeatureMap: TypeAlias = Annotated[torch.Tensor, Shape("B", "C", "H", "W")]

def encode(x: ImageBatch) -> FeatureMap:
    ...
```

TypeAliases may also be defined in a separate file and imported:

```python
# shapes.py
from typing import Annotated, TypeAlias
import torch
from torchshapeflow import Shape

ImageBatch: TypeAlias = Annotated[torch.Tensor, Shape("B", 3, "H", "W")]
```

```python
# model.py
from shapes import ImageBatch

def preprocess(x: ImageBatch):
    y = x.permute(0, 2, 3, 1)  # [B, H, W, 3] — inferred correctly
```

Both `X = Annotated[...]` (plain assignment) and `X: TypeAlias = Annotated[...]` (annotated assignment) are supported.

## Cross-file function calls

When a function in another file has tensor-annotated parameters and a tensor return annotation, calling it propagates the return shape. Symbolic dimensions from the parameter annotation are unified with the concrete shapes at the call site:

```python
# helpers.py
from typing import Annotated
import torch
from torchshapeflow import Shape

def embed(
    x: Annotated[torch.Tensor, Shape("B", "T")],
) -> Annotated[torch.Tensor, Shape("B", "T", 512)]:
    ...
```

```python
# main.py
from typing import Annotated
import torch
from torchshapeflow import Shape
from helpers import embed

def run(tokens: Annotated[torch.Tensor, Shape("B", "T")]):
    out = embed(tokens)  # inferred: [B, T, 512]
```

Same-file function calls with annotated signatures are supported too.

## Accepted tensor type expressions

The following base types are recognized as tensor annotations:

- `torch.Tensor`
- `Tensor` (after `from torch import Tensor`)
- Any qualified name ending in `.Tensor` (e.g. a custom subclass in `mylib.Tensor`)

## What `Shape` covers

`Shape` is the only semantic metadata object. There is no `DType`, `Device`, `Rank`, or `Layout` annotation — TorchShapeFlow is shape-only by design. See [Limitations](limitations.md).

## Behavior without `Shape`

| Annotation | Behavior |
|---|---|
| `Annotated[torch.Tensor, Shape(...)]` | Shape tracked |
| `Annotated[torch.Tensor, SomeOtherMeta()]` | Parse error — diagnostic emitted |
| `torch.Tensor` (no `Annotated`) | Silently ignored — no shape tracked |
| No annotation at all | Silently ignored |

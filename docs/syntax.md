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

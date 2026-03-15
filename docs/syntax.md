# Annotation Syntax

## Why annotate?

TorchShapeFlow follows the same philosophy as Pydantic: the value comes from
declaring contracts explicitly. Pydantic gives you runtime validation when you
declare field types; TorchShapeFlow gives you static shape verification when you
declare tensor shapes.

Shape checking is **opt-in per function**. The analyzer has nothing to check
until you add an annotation — and that is the point. You decide which boundaries
matter, annotate those parameters, and let the tool verify consistency from
there.

Symbolic dimensions like `"B"`, `"T"`, `"D"` are the primary mechanism.
In production deep-learning code, batch sizes and sequence lengths come from
config objects at runtime — not from literals in the source. The analyzer does
not try to chase config resolution. Instead, symbolic dims flow through
operations, and the analyzer verifies that every operation is consistent with the
declared shapes. Concrete integer dimensions (e.g. `3` for RGB channels, `768`
for a known embedding size) are a useful special case, not the default.

Start by annotating `forward` (or your main entry point), run `tsf check`, and
follow the diagnostics outward. See [Quickstart — Workflow](quickstart.md#workflow)
for a step-by-step guide.

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

Both `X = Annotated[...]` (plain assignment) and `X: TypeAlias = Annotated[...]` (annotated assignment) are supported. Python 3.12+ `type` statements work as well:

```python
type ImageBatch = Annotated[torch.Tensor, Shape("B", 3, "H", "W")]
```

### Project-level shape vocabulary

For larger projects, collect all shape aliases in a single file. This becomes
your project's shape vocabulary — a single source of truth for the tensor
contracts every module must respect:

```python
# shapes.py — project shape vocabulary
from typing import Annotated
import torch
from torchshapeflow import Shape

type ImageBatch   = Annotated[torch.Tensor, Shape("B", 3, "H", "W")]
type FeatureMap   = Annotated[torch.Tensor, Shape("B", "C", "H", "W")]
type TokenSequence = Annotated[torch.Tensor, Shape("B", "T")]
type Embedding    = Annotated[torch.Tensor, Shape("B", "T", "D")]
```

Other modules import from this file, keeping annotations short and consistent:

```python
# encoder.py
from shapes import ImageBatch, FeatureMap

def encode(x: ImageBatch) -> FeatureMap:
    ...
```

```python
# decoder.py
from shapes import FeatureMap, Embedding

def decode(features: FeatureMap) -> Embedding:
    ...
```

The analyzer resolves these aliases at analysis time, so shape inference and
diagnostics work exactly as if the full `Annotated[...]` form were written
inline.

### Importing TypeAliases from other files

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

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

If you want to run the example PyTorch scripts themselves, install the examples
extra as well:

```bash
uv sync --extra dev --extra examples
```

## Annotate your tensors

TorchShapeFlow reads `Annotated[torch.Tensor, Shape(...)]` contracts from your
annotations. In practice, that usually means function parameters first, then
shared shape aliases and annotated local variables where you want a stronger
contract inside the function body. String dimensions are symbolic; integer
dimensions are constant. In real config-driven model code, symbolic dimensions
are the default path. Use integer dimensions when an axis is genuinely fixed by
the contract, such as RGB channels or a known embedding width:

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
All clean (1 file checked)
```

**With errors:**

```
broken.py:9:9 error TSF1004 Invalid reshape.
broken.py:17:12 warning TSF1006 Broadcasting incompatibility.
1 error, 1 warning in 1 file (1 file checked)
```

Exit code is `0` when no errors are found, `1` otherwise.

Use `--verbose` (or `-v`) to see per-file status for clean files:

```bash
tsf check src/ --verbose
```

```
mymodel.py: ok
utils.py: ok
All clean (2 files checked)
```

## Try the bundled examples

```bash
tsf check examples/simple_cnn.py
tsf check examples/transformer_block.py
tsf check examples/error_cases.py --json
```

## Workflow

TorchShapeFlow is opt-in: it only checks functions whose parameters carry
`Annotated[torch.Tensor, Shape(...)]` annotations. A practical adoption path:

1. **Start with `forward`** — annotate the main entry point of your module. This
   is where input shapes are known and where most shape bugs surface.

2. **Run `tsf check`** — the analyzer will propagate shapes through the
   operations in that function and report any mismatches.

3. **Follow the warnings** — diagnostics point to operations where a shape
   could not be verified. These are often calls to helper functions that lack
   annotations. Add annotations there next.

4. **Define a shape vocabulary** — once you have several annotated functions,
   extract common shapes into a shared `shapes.py` using `TypeAlias` (see
   [Type alias pattern](syntax.md#type-alias-pattern)). This keeps annotations
   short and consistent across your codebase.

5. **Annotate helper functions** — adding parameter and return annotations to
   helpers enables cross-function shape inference. The analyzer unifies symbolic
   dimensions at each call site, catching mismatches that span module
   boundaries.

Each step is incremental — you get value from the first annotation, and coverage
grows as you add more.

## Getting annotation proposals (`tsf suggest`)

Once you have annotated parameters, the analyzer often already knows the
return shape. Run:

```bash
tsf suggest path/to/mymodel.py
```

…and TorchShapeFlow emits JSON proposals for return annotations it can
already verify, without touching your source. Example output:

```json
{
  "files": [
    {
      "path": "model.py",
      "suggestions": [
        {
          "line": 6, "column": 5,
          "function": "scores",
          "shape": "[B, H, T, T]",
          "annotation": "Annotated[torch.Tensor, Shape(\"B\", \"H\", \"T\", \"T\")]",
          "kind": "return_annotation"
        }
      ]
    }
  ]
}
```

Review each suggestion and paste it into your function definition. TSF
**never writes suggestions back** — it proposes; you (or your editor/agent)
decide.

Suggestions are emitted only when every precondition holds:

- At least one parameter has a `Shape` annotation (you opted in).
- The function has no return annotation yet.
- Every exit path provably returns a value. Recognized terminators are a
  trailing `return X`, a trailing `raise`, and `if/else` where every
  branch terminates. Loops, `try/except`, `match`, and bare `return` are
  treated as "don't know" and silence the suggestion.
- Every `return` statement produces a tensor with the same shape.
- Every dimension is expressible in `Shape(...)` syntax (symbolic names
  and integer constants).
- The first annotated parameter uses an inline `Annotated[..., Shape(...)]`
  or `Annotated[..., "B T D"]` spelling — the suggestion reuses its form
  so the proposed annotation refers only to names the file already
  imports. `TypeAlias` params skip the suggestion.

Anything outside this envelope is silently skipped — a function without a
suggestion is not an error.

## Next steps

- [Annotation syntax](syntax.md) — all supported annotation forms
- [Supported operators](operators.md) — what shapes are tracked and how
- [Limitations](limitations.md) — what the analyzer does not handle

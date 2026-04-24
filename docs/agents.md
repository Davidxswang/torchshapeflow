# For AI Coding Agents

This page is for AI coding agents (Claude Code, Cursor, Copilot, Aider, or any
tool-using LLM) that edit PyTorch code. If you are a human looking for a
tutorial, read the [Quickstart](quickstart.md) instead.

## What TorchShapeFlow does

TSF is a static AST-based shape analyzer. It reads annotations of the form
`Annotated[torch.Tensor, Shape(...)]` on function parameters and verifies that
the operations in the body are consistent with those shapes. No code is
executed; no config objects are resolved; no on-disk tensor data is inspected.
This is by design — see [Annotation Syntax — Why annotate?](syntax.md#why-annotate).

If you want a shape bug caught, the user (or you) must declare the shape
contract on the function boundary. The analyzer then propagates symbolic
dimensions through supported operators and reports mismatches.

## When to reach for TSF

- You are writing or modifying a PyTorch `nn.Module.forward` method, a model
  component, or any function that operates on tensors.
- You are about to paste code that does `@`, `matmul`, `reshape`, `view`,
  `permute`, or passes tensors through `nn.Linear` / `nn.Conv2d` / `nn.LSTM` —
  these are the failure modes TSF is best at catching.
- The user asks for a shape-correctness check, or for the return type of a
  function they have partially annotated.

If the code has no `Annotated[..., Shape(...)]` annotations anywhere, TSF will
be silent. That is not a bug — it is the design. In that case, add a minimal
annotation first (see [Minimal annotation recipe](#minimal-annotation-recipe)).

## Two commands

### `tsf check --json <path>`

Runs the analyzer and emits diagnostics + hover facts.

- **Exit 0** means every analyzed file is clean (no `error`-severity
  diagnostics).
- **Exit 1** means at least one file produced an error-severity diagnostic.
  Warnings alone do not flip the exit code.
- The JSON payload is `{ "files": [ { path, diagnostics, hovers } ] }`.

Shape-mismatch diagnostics (TSF1003, TSF1007, etc.) carry structured
`expected` / `actual` / `hint` fields alongside the prose `message`. Prefer
the structured fields when deciding how to fix the code; the prose is for
surfacing to a human. See [Architecture — Diagnostic JSON schema](architecture.md#diagnostic-json-schema).

### `tsf suggest <path>`

Proposes annotations the analyzer has already verified — today, only return
annotations for functions whose parameters are annotated and whose every exit
path returns a shape-stable tensor.

- **Exit 0** means analysis succeeded (regardless of whether suggestions were
  produced).
- **Exit 1** means at least one file emitted an error-severity diagnostic.
  This matters: an empty `suggestions` list alongside a clean exit means "TSF
  verified the function and has nothing to add." An empty `suggestions` list
  alongside a non-zero exit means "TSF could not verify some function — read
  the `diagnostics` to see why." Do not conflate the two.
- Payload is `{ "files": [ { path, diagnostics, suggestions } ] }`.
- The command is read-only. It never rewrites source files. If you act on a
  suggestion, you do the edit; TSF only recommends.

Each suggestion includes a pasteable `annotation` string and a raw `shape`.
The annotation is rendered by reusing the first annotated parameter's AST as
a template, so names like `Annotated`, `Tensor`, `torch.Tensor`, and
`typing.Annotated` always match what the target file already imports. Paste
the `annotation` verbatim.

## Decision tree

```
User edited PyTorch code?
├── yes → run tsf check --json
│         ├── exit 1 → surface diagnostics to user; for each TSF1xxx error,
│         │           the `expected`, `actual`, and `hint` fields tell you
│         │           exactly what to change
│         └── exit 0 → (optional) run tsf suggest to see whether TSF can
│                     fill in return annotations you have not written yet
└── no  → no action needed
```

## Minimal annotation recipe

If the user's code has no annotations, the smallest first step is:

```python
from typing import Annotated
import torch
from torchshapeflow import Shape


def forward(
    self,
    x: Annotated[torch.Tensor, Shape("B", "T", 768)],
):
    ...
```

Symbolic names (`"B"`, `"T"`) are the default. Use integer constants only for
axes that are semantically fixed (`3` for RGB, `768` for a known embedding
width). Dimensions that come from config objects or from disk-loaded data
should stay symbolic — the analyzer cannot see their values.

For larger projects, centralize shape aliases in a `shapes.py` module. See
[Annotation Syntax — Type alias pattern](syntax.md#type-alias-pattern).

## Failure modes to recognize

| Symptom | Likely cause | What to do |
|---|---|---|
| `tsf check` emits no diagnostics and no hovers | No `Shape` annotations in the file | Add one annotation to a function parameter and re-run |
| `TSF2001` / `TSF2002` / `TSF2003` warnings | Inference lost at a specific operator or call site | Either the operator is unsupported (fine to ignore) or you can annotate the called helper to restore tracking |
| `tsf suggest` returns empty `suggestions` with exit 0 | TSF analyzed the function cleanly but one of the preconditions did not hold (e.g. multiple return paths with different shapes, a generator, an ExpressionDim in the inferred shape) | No action required — the feature is intentionally narrow |
| `tsf suggest` returns empty `suggestions` with exit 1 | An error-severity diagnostic fired; the function is not endorsed | Read `diagnostics`, fix the reported error, then re-run |

## What TSF will not do

- Execute user code to infer shapes.
- Resolve config objects like `cfg.d_model` or YAML entries.
- Infer shapes of tensors loaded from disk (`torch.load`, dataset `__getitem__`,
  HDF5 / Parquet / pickle).
- Rewrite source files.
- Check dtype, device, layout, or distributed-tensor semantics — it is
  shape-only by design.

See [Limitations](limitations.md) for the complete non-goals list.

---
name: torchshapeflow
description: Verify and propose PyTorch tensor shape annotations. Use when the user is editing code that performs tensor operations (`@`, `matmul`, `reshape`, `view`, `permute`, `nn.Linear`, `nn.Conv2d`, `nn.LSTM`), writes an `nn.Module.forward` method, or asks about tensor shapes / shape bugs. The skill exposes three MCP tools (`check`, `suggest`, `hover_at`) that run a static, AST-based shape analyzer — no execution required.
---

# TorchShapeFlow for Agents

This skill drives the `torchshapeflow` MCP server. You have three tools:

| Tool | Purpose |
|---|---|
| `check(path)` | Run shape analysis on a file or directory. Returns `{files: [{path, diagnostics, hovers}]}`. |
| `suggest(path)` | Propose return annotations TSF has already verified. Returns `{files: [{path, diagnostics, suggestions}]}`. Read-only — TSF never writes source. |
| `hover_at(path, line, column)` | Return the inferred shape at a 1-based source location. |

## Core mental model

TorchShapeFlow is **annotation-first**, like Pydantic. It checks shapes only where the user has written a contract:

```python
def forward(x: Annotated[torch.Tensor, Shape("B", "T", 768)]) -> Annotated[torch.Tensor, Shape("B", "T", 256)]:
    ...
```

Symbolic dimensions (`"B"`, `"T"`, `"D"`) are the default path. Integer constants (`3`, `768`) are for semantically fixed axes. **Config-driven sizes and disk-loaded tensors stay symbolic** — the analyzer cannot see runtime values, and attempting to chase them silently invalidates the contract.

If the file has no `Annotated[..., Shape(...)]` annotations, TSF will be silent. That is not a bug. The first step in that case is to add a minimal parameter annotation.

## When to reach for each tool

- **After writing or editing PyTorch code** → call `check` on the file. If the user hasn't annotated any parameters yet, suggest the minimal annotation pattern below and then call `check` once they accept.
- **When the user has annotated parameters but no return annotation** → call `suggest`. If it returns a non-empty `suggestions` list, offer the pasteable `annotation` string. TSF never writes source; you apply the edit.
- **When the user asks about a specific tensor** (e.g. "what shape is `q` here?") → call `hover_at` with the file + 1-based line/column.

The post-edit hook will also run `check` automatically after `Write`/`Edit` on any `*.py` file and surface errors to the session. You do not need to re-run `check` immediately after an edit — the hook already did.

## Reading diagnostics

`check` returns per-file `diagnostics`. Each diagnostic has:

- `code` — stable TSF identifier (`TSF1003`, `TSF1007`, …).
- `severity` — `"error"` or `"warning"`.
- `message` — human-readable summary.
- `path`, `line`, `column` — 1-based location.

Shape-mismatch diagnostics also carry three **structured fields** you should prefer over parsing the prose:

- `expected` — what the analyzer required (e.g. `"last dim = 768"`).
- `actual` — what it saw (e.g. `"[B, T, 512] (last dim = 512)"`).
- `hint` — a concrete suggested fix.

Apply the `hint` directly when it names the fix. Example:

```
"expected": "last dim = 768",
"actual":   "[B, T, 512] (last dim = 512)",
"hint":     "change nn.Linear(in_features=...) to 512, or reshape the input so its last dim equals 768"
```

## Exit-code semantics (for interpreting tool output)

- `check`: empty `diagnostics` → file is clean. Any entry with `severity: "error"` means the user's code has a shape bug you should fix.
- `suggest`: empty `suggestions` + empty `diagnostics` → TSF verified the function and has nothing to add. Empty `suggestions` + non-empty `diagnostics` → TSF could not verify the function; read the diagnostics and fix them first, don't paste any proposal.
- `hover_at`: `null` → no tracked shape at that position. Tell the user the analyzer has no information there rather than guessing.

## Minimal annotation recipe

If the user's file has no annotations yet, this is the smallest useful first step:

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

Guidelines:

- Symbolic names (`"B"`, `"T"`) by default.
- Integer constants only when an axis is semantically fixed (`3` for RGB, `768` for a known embedding width, etc.).
- For larger projects, centralize shape aliases in a `shapes.py` module:
  ```python
  from typing import Annotated, TypeAlias
  import torch
  from torchshapeflow import Shape

  ImageBatch: TypeAlias = Annotated[torch.Tensor, Shape("B", 3, "H", "W")]
  ```

## Failure modes

| Symptom | Likely cause | What to do |
|---|---|---|
| `check` returns empty `diagnostics` AND empty `hovers` | The file has no `Shape` annotations | Propose the minimal annotation on one parameter, then re-run |
| `TSF2001` / `TSF2002` / `TSF2003` warnings | Shape tracking lost at a specific operator or call site | Either the operator is unsupported (acknowledge and move on) or you can annotate the called helper to restore tracking |
| `suggest` returns empty `suggestions` with exit clean | TSF cleared the function but a precondition did not hold (multiple return paths with different shapes, generator function, ExpressionDim in the inferred shape, param uses a TypeAlias) | No action required — the feature is intentionally narrow |
| `suggest` returns empty `suggestions` with error diagnostics | The function has a shape bug; TSF refuses to propose for broken code | Fix the reported error first, then re-invoke |

## What TSF will not do

TorchShapeFlow is static and shape-only. It will not:

- Execute user code to infer shapes.
- Resolve config objects (`cfg.d_model`, CLI flags, YAML).
- Infer shapes of tensors loaded from disk (`torch.load`, dataset `__getitem__`, HDF5 / Parquet / pickle).
- Rewrite source files.
- Check dtype, device, layout, or distributed-tensor semantics.

Do not ask it to. When the shape genuinely cannot be known statically, leave the dim symbolic and move on.

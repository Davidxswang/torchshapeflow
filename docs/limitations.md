# Limitations

TorchShapeFlow is a static analyzer. It does not execute code and is intentionally narrow in scope. When inference is not possible, it degrades gracefully — returning `None` for the shape and optionally emitting a diagnostic — rather than crashing.

## Control flow is partially tracked

`if`/`else` blocks are analyzed: both branches are walked and the resulting shape environments are merged. Dimensions that agree are kept; dimensions that differ become `?` (unknown). `for`, `while`, `try/except`, and `match` are not analyzed.

## Only some Python constructs are supported

The following are not handled:

- List/dict/set comprehensions
- Generator expressions
- Closures and nested functions
- Decorators
- `*args` / `**kwargs` unpacking
- `getattr` and dynamic attribute access

## Only listed operators produce inferred shapes

When an unsupported call is encountered while shape-tracked tensors are in play,
TorchShapeFlow emits inference gap warnings (TSF2001, TSF2002, TSF2003) so
that users know where shape tracking is lost. Outside functions where no shape
contracts are active, unsupported calls are silently ignored. See
[Supported Operators](operators.md) for the full list.

## dtype, device, and layout are not analyzed

TorchShapeFlow is shape-only by design. `DType`, `Device`, `Layout`, and distributed tensor semantics are out of scope.

## Known analyzer limitations

- **Cross-file resolution is project-relative only.** Imports are resolved relative to the importing file's directory. Third-party packages (e.g. `from torch.nn import Linear`) are not indexed — only project-local `.py` files.
- **Symbolic input dims prioritize propagation over fixed-constant validation.**
  `Shape("B", "C", "H", "W")` through `nn.Conv2d(3, 64, 3)` will not verify
  `C == 3`, but the output `[B, 64, H_out, W_out]` is still inferred. Likewise
  for `nn.Linear`. This is intentional: symbolic contracts are the primary path
  for config-driven code, while constant validation is an extra check when a
  dimension is known statically.
- **Non-literal constructor args fall back to `__init__` defaults.** If a constructor arg is a name (e.g. `nn.Linear(in_dim, out_dim)`) and the corresponding `__init__` parameter has an integer default, that default is used. If there is no default, the spec is dropped and no shape is inferred.

## Explicit non-goals for the MVP

- Executing user code to infer shapes
- Runtime instrumentation or tracing
- TorchScript, `torch.compile`, or `torch.fx` integration
- Full language-server functionality
- Auto-fixing user code
- Perfect soundness for highly dynamic code

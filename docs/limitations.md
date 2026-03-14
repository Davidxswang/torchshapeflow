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

Unsupported calls produce no diagnostic — the result is silently dropped from the shape environment. See [Supported Operators](operators.md) for the full list.

## dtype, device, and layout are not analyzed

TorchShapeFlow is shape-only by design. `DType`, `Device`, `Layout`, and distributed tensor semantics are out of scope.

## Known analyzer limitations

- **Cross-file resolution is project-relative only.** Imports are resolved relative to the importing file's directory. Third-party packages (e.g. `from torch.nn import Linear`) are not indexed — only project-local `.py` files.
- **Symbolic input dim skips validation but still propagates output.** `Shape("B", "C", "H", "W")` through `nn.Conv2d(3, 64, 3)` will not verify `C == 3`, but the output `[B, 64, H_out, W_out]` is still inferred. Likewise for `nn.Linear`.
- **Non-literal constructor args fall back to `__init__` defaults.** If a constructor arg is a name (e.g. `nn.Linear(in_dim, out_dim)`) and the corresponding `__init__` parameter has an integer default, that default is used. If there is no default, the spec is dropped and no shape is inferred.

## Explicit non-goals for the MVP

- Executing user code to infer shapes
- Runtime instrumentation or tracing
- TorchScript, `torch.compile`, or `torch.fx` integration
- Full language-server functionality
- Auto-fixing user code
- Perfect soundness for highly dynamic code

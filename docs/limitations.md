# Limitations

TorchShapeFlow is a static analyzer. It does not execute code and is intentionally narrow in scope. When inference is not possible, it degrades gracefully — returning `None` for the shape and optionally emitting a diagnostic — rather than crashing.

## Control flow is not tracked

`if`/`else`, `for`, `while`, `try/except`, and `match` are not analyzed. Only straight-line code within a function body is propagated. Shapes defined inside a branch are not visible after it.

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
- **Non-literal `out_channels` / `out_features` drops the spec entirely.** If the output-size constructor arg is non-literal (e.g. `nn.Linear(512, cfg.hidden)`), the spec cannot be collected and no shape is inferred. Non-literal *input*-size args (`in_channels`, `in_features`) are tolerated — the input-dim check is skipped and the output shape is still propagated.
- **Symbolic input dim skips validation but still propagates output.** `Shape("B", "C", "H", "W")` through `nn.Conv2d(3, 64, 3)` will not verify `C == 3`, but the output `[B, 64, H_out, W_out]` is still inferred. Likewise for `nn.Linear`.
- **No symbolic unification.** `"B"` in one annotation and `"B"` in another are independent — there is no constraint propagation across call sites.
- **The `-1` reshape dim is not validated for mixed shapes.** The inferred dimension is computed as a quotient expression, but product consistency is only checked when all input and requested dims are constant.
- **Open-ended slice sizes are not tracked.** `x[1:]` and `x[:n]` (where `n` is a variable) keep the dimension unchanged. Only slices with two explicit constant integer bounds (e.g. `x[1:5]` → 4) are tracked.

## Explicit non-goals for the MVP

- Executing user code to infer shapes
- Runtime instrumentation or tracing
- TorchScript, `torch.compile`, or `torch.fx` integration
- Full language-server functionality
- Auto-fixing user code
- Perfect soundness for highly dynamic code

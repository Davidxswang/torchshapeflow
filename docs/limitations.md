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
- Augmented assignment (`x += y`)

## Only listed operators produce inferred shapes

Unsupported calls produce no diagnostic — the result is silently dropped from the shape environment. See [Supported Operators](operators.md) for the full list.

## dtype, device, and layout are not analyzed

TorchShapeFlow is shape-only by design. `DType`, `Device`, `Layout`, and distributed tensor semantics are out of scope.

## Single-file analysis only

Each file is analyzed independently. Imported functions and classes defined in other files are not resolved. If `forward` calls a helper defined in another module, that helper's output has no tracked shape.

## Known analyzer limitations

- **Module aliases are not tracked.** Only direct `self.attr` access is resolved. `m = self.linear; m(x)` is not supported — `m` has no tracked shape spec.
- **Symbolic last dim in `nn.Linear`.** `in_features` matching requires a `ConstantDim`. A symbolic last dim (e.g. `"D"`) will not match against `in_features=768` even if semantically intended.
- **Symbolic channel dim in `nn.Conv2d`.** The channel dimension must be a `ConstantDim` equal to `in_channels`.
- **No symbolic unification.** `"B"` in one annotation and `"B"` in another are independent — there is no constraint propagation across call sites.
- **The `-1` reshape dim is not validated for mixed shapes.** The inferred dimension is computed as a quotient expression, but product consistency is only checked when all input and requested dims are constant.
- **Slice sizes are not tracked.** `x[1:5]` keeps the dimension but does not infer its size as `4`. The size remains whatever it was before indexing.

## Explicit non-goals for the MVP

- Executing user code to infer shapes
- Runtime instrumentation or tracing
- TorchScript, `torch.compile`, or `torch.fx` integration
- Full language-server functionality
- Auto-fixing user code
- Perfect soundness for highly dynamic code

# Limitations

TorchShapeFlow is a static analyzer. It does not execute code and is intentionally narrow in scope. When inference is not possible, it degrades gracefully — returning `None` for the shape and optionally emitting a diagnostic — rather than crashing.

## Control flow is partially tracked

`if`/`else` blocks are analyzed: both branches are walked and the resulting shape environments are merged. Dimensions that agree are kept; dimensions that differ become `?` (unknown). `while`, `try/except`, and `match` are not analyzed. `for` loops are not analyzed in general, except for a narrow `__init__` pattern used to summarize loop-built `nn.Sequential(*layers)` stacks.

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

- **Cross-file resolution is project-relative only.** Imports are resolved relative to the importing file's directory. Project-local aliases, annotated helper functions, and annotated custom-module `forward()` contracts can be imported and reused. Third-party packages (e.g. `from torch.nn import Linear`) are not indexed — only project-local `.py` files.
- **Custom module support is annotation-driven.** TorchShapeFlow only tracks a custom `nn.Module` when its `forward()` method has tensor shape annotations. Constructor internals are not analyzed generically; the shape contract comes from the annotated `forward()` signature.
- **Symbolic input dims emit a hint when a constant is required.**
  `Shape("B", "C", "H", "W")` through `nn.Conv2d(3, 64, 3)` cannot verify
  `C == 3`, but output `[B, 64, H_out, W_out]` is still inferred and a
  `TSF1012` warning is emitted suggesting the user replace `C` with `3` in
  their annotation. If `C` truly must equal 3, annotating it as a constant
  makes the contract explicit and enables hard mismatch detection. Likewise
  for `nn.Linear.in_features` and `nn.LSTM.input_size`.
- **Variable constructor args produce symbolic output dims.** If an output dimension constructor arg is a variable name (e.g. `nn.Linear(in_dim, out_dim)`), the variable name itself is used as a symbolic label — `out_dim` becomes `SymbolicDim("out_dim")` in the inferred shape. This is always correct regardless of what the caller passes. Literal integer args produce exact `ConstantDim` values as usual.

## Explicit non-goals for the MVP

- Executing user code to infer shapes
- Runtime instrumentation or tracing
- Inferring shapes that only exist at runtime — config-driven sizes and
  disk-loaded data. See
  [Annotation Syntax — Why annotate?](syntax.md#why-annotate) for the rationale
  (this is the reason symbolic dims are the primary mechanism).
- TorchScript, `torch.compile`, or `torch.fx` integration
- Full language-server functionality
- Auto-fixing user code
- Perfect soundness for highly dynamic code

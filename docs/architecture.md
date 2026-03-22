# Architecture

## Analysis pipeline

TorchShapeFlow analyzes a target file in one statement-by-statement walk, with
optional project-local indexing to resolve imported aliases and annotated helper
signatures:

1. **Parse** — `ast.parse` converts source text into an AST module (`parser.parse_source`).
2. **Resolve aliases and function signatures** — file-level type aliases are collected from `X = Annotated[...]`, `X: TypeAlias = Annotated[...]`, and, on Python 3.12+ runtimes, `type X = Annotated[...]`. If a `ProjectIndex` is present, project-local `from ... import ...` references are resolved first so imported aliases and annotated helper signatures can be used during analysis. Inside function bodies, local aliases declared with the same forms are added to the local alias scope from the point where they appear.
3. **Collect module specs** — `_collect_class_specs` walks class `__init__` bodies to find `nn.Linear`, `nn.Conv2d`, `nn.Embedding`, `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.Sequential`, `nn.MultiheadAttention`, and passthrough module assignments, recording their constructor arguments as spec values.
4. **Seed shape environment** — for each function (or `forward` method), annotated parameters are parsed via `parser.parse_tensor_annotation` and added to the environment `env: dict[str, Value]`.
5. **Propagate shapes** — `_analyze_statement` walks the function body statement by statement. For each assignment, `_eval_expr` evaluates the right-hand side, dispatching to the appropriate rule function. Local annotated variable declarations are treated as shape contracts, and local alias declarations update the in-scope alias table for later statements. Results are stored back into `env`.
6. **Emit results** — diagnostics and hover facts accumulate in a `ModuleContext` and are returned as a `FileReport`.

## Module map

| Module | Responsibility |
|---|---|
| `model.py` | All core data types (`Dim` variants, `TensorShape`, `TensorValue`, `TensorTupleValue`, `LinearSpec`, `Conv2dSpec`, `PassthroughSpec`, `EmbeddingSpec`, `Pool2dSpec`, `SequentialSpec`, `MultiheadAttentionSpec`, `ModuleSpec`, `Value`). Shape arithmetic: `product_dim`, `quotient_dim`, `sum_dim`, `broadcast_shapes`, `batch_matmul_shape`, `normalize_index`. |
| `annotations.py` | Public `Shape` class used in `Annotated[Tensor, Shape(...)]`. |
| `parser.py` | Parses `Annotated[Tensor, Shape(...)]` annotation AST nodes into `TensorValue`. Raises `AnnotationParseError` on malformed annotations. |
| `analyzer.py` | Main AST walker. Manages the shape environment, dispatches to rule functions, emits diagnostics via `ModuleContext`. |
| `index.py` | Project-local alias and annotated-function indexing (`ProjectIndex`, `FuncSig`, symbolic substitution for cross-file calls). |
| `diagnostics.py` | `Diagnostic` dataclass and `Severity` type alias (`"error" \| "warning"`). |
| `report.py` | `FileReport` (list of diagnostics + hover facts per file) and `HoverFact` (inferred shape at a source location). |
| `cli.py` | Typer CLI. `tsf check` runs the analyzer and formats output. `tsf version` prints the package version. |
| `rules/__init__.py` | Re-exports all public inference functions. |
| `rules/shape_ops.py` | Tensor/functional shape-operator inference. See [Supported Operators](operators.md) for the canonical user-facing inventory. |
| `rules/broadcasting.py` | `infer_binary_broadcast` — wraps `broadcast_shapes` for element-wise ops. |
| `rules/linear.py` | `infer_linear` for `nn.Linear`. |
| `rules/conv2d.py` | `infer_conv2d` for `nn.Conv2d`. |
| `rules/embedding.py` | `infer_embedding` for `nn.Embedding`. |
| `rules/pool2d.py` | `infer_pool2d` for `nn.MaxPool2d` and `nn.AvgPool2d`. |
| `rules/indexing.py` | `infer_subscript` for tensor subscript and shape-tuple indexing. |
| `rules/common.py` | Shared AST helpers: `int_from_ast`, `qualified_name`, `dim_from_value`, `tuple_index`, `spatial_output_dim`. |
| `utils/paths.py` | `collect_python_files` — recursive `.py` file discovery. |

## Dim type hierarchy

```
Dim  (TypeAlias)
├── ConstantDim(value: int)       — a fixed integer size, e.g. 32
├── SymbolicDim(name: str)        — a named unknown size, e.g. "B"
├── ExpressionDim(expr: str)      — a derived expression, e.g. "4*B" or "(B*C)/4"
└── UnknownDim(token: str)        — explicitly unresolvable
```

Shape arithmetic returns `ConstantDim` when all operands are constant and `ExpressionDim` otherwise. Expressions are stored as strings and compared structurally.

## Shape environment

The environment `env: dict[str, Value]` maps variable names to their inferred `Value`:

```
Value  (TypeAlias)
├── TensorValue(shape: TensorShape, origin: str | None)
├── ShapeTupleValue(dims: tuple[Dim, ...])   — result of x.shape or x.size()
├── IntegerValue(value: int | None)           — result of x.ndim or x.size(i)
├── TensorTupleValue(tensors: tuple[TensorValue, ...])  — result of chunk/split/MHA
│
│   ModuleSpec  (TypeAlias — stored in module_specs and env)
├── LinearSpec(in_features, out_features)     — nn.Linear
├── Conv2dSpec(in_channels, out_channels, kernel_size, stride, padding, dilation)  — nn.Conv2d
├── PassthroughSpec()                         — shape-preserving modules (BatchNorm, ReLU, …)
├── EmbeddingSpec(embedding_dim)              — nn.Embedding
├── Pool2dSpec(kernel_size, stride, padding, dilation)  — nn.MaxPool2d / nn.AvgPool2d
├── SequentialSpec(specs: tuple[ModuleSpec, ...])       — nn.Sequential
└── MultiheadAttentionSpec(embed_dim, num_heads, batch_first)  — nn.MultiheadAttention
```

Spec values are stored in `module_specs` (keyed by attribute name) when their constructor is parsed from `__init__`. When `self.linear(x)` is called, the analyzer looks up `"linear"` in `module_specs`, retrieves the spec, and calls the appropriate inference function. Module aliases (`m = self.linear; m(x)`) are also supported: spec values stored in `env` are looked up before falling through to `func_sigs`.

When an annotated function call is resolved through `func_sigs`, symbolic
dimensions in the callee signature are unified with the caller argument shapes
and substituted into the declared return shape. Imported helper functions are
handled the same way when they can be resolved through `ProjectIndex`.

## Diagnostic codes

| Code | Severity | Trigger |
|---|---|---|
| `TSF1001` | error | Annotation parse error (malformed `Annotated` or `Shape`) |
| `TSF1002` | — | Reserved (not used) |
| `TSF1003` | error | Incompatible `matmul` / `bmm` shapes |
| `TSF1004` | error | Invalid `reshape` or `flatten` dimensions |
| `TSF1005` | error | Invalid `cat` or `stack` dimensions or mismatched shapes |
| `TSF1006` | error or warning | Broadcasting incompatibility (error when both dims are constant; warning when one or both are symbolic) |
| `TSF1007` | error | `nn.Linear`, `nn.Conv2d`, or `nn.MaxPool2d`/`AvgPool2d` input shape mismatch |
| `TSF1008` | error | Invalid `permute`, `transpose`, `squeeze`, `unsqueeze`, `chunk`, or `movedim` dimensions |
| `TSF1009` | error | Return shape does not match the declared return type annotation |
| `TSF1010` | error | Symbolic dim bound to conflicting values across call-site arguments |
| `TSF1011` | error | Local annotated variable shape does not match the inferred shape |
| `TSF1012` | warning | Symbolic dim used where a specific constant is required (e.g. `nn.Linear.in_features`, `nn.Conv2d.in_channels`) — suggests replacing with the literal constant in the `Shape` annotation |
| `TSF2001` | warning | Unsupported tensor method or unresolvable method arguments — shape inference lost |
| `TSF2002` | warning | Call to unannotated function with tensor arg — shape inference lost |
| `TSF2003` | warning | Unresolvable module `self.xxx` — no spec inferred |

## Adding a new operator

See [Development → Adding a new operator](development.md#adding-a-new-operator).
Operator behavior and support status should be documented only once in
[Supported Operators](operators.md); this page describes the implementation
structure, not the canonical support matrix.

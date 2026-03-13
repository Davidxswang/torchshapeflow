# Architecture

## Analysis pipeline

TorchShapeFlow analyzes Python source in a single pass per file:

1. **Parse** — `ast.parse` converts source text into an AST module (`parser.parse_source`).
2. **Collect module specs** — `_collect_class_specs` walks class `__init__` bodies to find `nn.Linear` and `nn.Conv2d` assignments, recording their constructor arguments as `LinearSpec` / `Conv2dSpec` values.
3. **Seed shape environment** — for each function (or `forward` method), annotated parameters are parsed via `parser.parse_tensor_annotation` and added to the environment `env: dict[str, Value]`.
4. **Propagate shapes** — `_analyze_statement` walks the function body statement by statement. For each assignment, `_eval_expr` evaluates the right-hand side, dispatching to the appropriate rule function. Results are stored back into `env`.
5. **Emit results** — diagnostics and hover facts accumulate in a `ModuleContext` and are returned as a `FileReport`.

## Module map

| Module | Responsibility |
|---|---|
| `model.py` | All core data types (`Dim` variants, `TensorShape`, `TensorValue`, `LinearSpec`, `Conv2dSpec`, `Value`). Shape arithmetic: `product_dim`, `quotient_dim`, `sum_dim`, `broadcast_shapes`, `batch_matmul_shape`, `normalize_index`. |
| `annotations.py` | Public `Shape` class used in `Annotated[Tensor, Shape(...)]`. |
| `parser.py` | Parses `Annotated[Tensor, Shape(...)]` annotation AST nodes into `TensorValue`. Raises `AnnotationParseError` on malformed annotations. |
| `analyzer.py` | Main AST walker. Manages the shape environment, dispatches to rule functions, emits diagnostics via `ModuleContext`. |
| `diagnostics.py` | `Diagnostic` dataclass and `Severity` type alias (`"error" \| "warning" \| "hint"`). |
| `report.py` | `FileReport` (list of diagnostics + hover facts per file) and `HoverFact` (inferred shape at a source location). |
| `cli.py` | Typer CLI. `tsf check` runs the analyzer and formats output. `tsf version` prints the package version. |
| `rules/__init__.py` | Re-exports all public inference functions. |
| `rules/shape_ops.py` | `infer_permute`, `infer_transpose`, `infer_reshape`, `infer_flatten`, `infer_squeeze`, `infer_unsqueeze`, `infer_size`, `infer_cat`, `infer_stack`, `infer_matmul`. |
| `rules/broadcasting.py` | `infer_binary_broadcast` — wraps `broadcast_shapes` for element-wise ops. |
| `rules/linear.py` | `infer_linear` for `nn.Linear`. |
| `rules/conv2d.py` | `infer_conv2d` for `nn.Conv2d`. |
| `rules/indexing.py` | `infer_subscript` for tensor subscript and shape-tuple indexing. |
| `rules/common.py` | Shared AST helpers: `int_from_ast`, `qualified_name`, `dim_from_value`, `tuple_index`. |
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
├── LinearSpec(in_features, out_features)     — collected from __init__
└── Conv2dSpec(in_channels, out_channels, kernel_size, stride, padding, dilation)
```

`LinearSpec` and `Conv2dSpec` are stored in the environment when their constructor is parsed from `__init__`. When `self.linear(x)` is called, the analyzer looks up `"linear"` in `module_specs`, retrieves the `LinearSpec`, and calls `infer_linear`.

## Diagnostic codes

| Code | Trigger |
|---|---|
| `TSF1001` | Annotation parse error (malformed `Annotated` or `Shape`) |
| `TSF1003` | Incompatible `matmul` / `bmm` shapes |
| `TSF1004` | Invalid `reshape` or `flatten` dimensions |
| `TSF1005` | Invalid `cat` or `stack` dimensions or mismatched shapes |
| `TSF1006` | Broadcasting incompatibility |
| `TSF1007` | `nn.Linear` or `nn.Conv2d` input shape mismatch |
| `TSF1008` | Invalid `permute`, `transpose`, `squeeze`, or `unsqueeze` dimensions |

## Adding a new operator

See [Development → Adding a new operator](development.md#adding-a-new-operator).

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from torchshapeflow.diagnostics import Diagnostic, Severity
from torchshapeflow.index import (
    FuncSig,
    ProjectIndex,
    apply_substitution,
    collect_imports,
    collect_raw_aliases,
    extract_func_sig,
    resolve_aliases,
    unify_dims,
)
from torchshapeflow.model import (
    ConstantDim,
    Conv2dSpec,
    Dim,
    EmbeddingSpec,
    ExpressionDim,
    IntegerValue,
    LinearSpec,
    ModuleSpec,
    MultiheadAttentionSpec,
    PassthroughSpec,
    Pool2dSpec,
    SequentialSpec,
    ShapeTupleValue,
    SymbolicDim,
    TensorShape,
    TensorTupleValue,
    TensorValue,
    UnknownDim,
    Value,
    broadcast_has_uncertain_dims,
    make_dim,
    normalize_index,
    render_dim,
)
from torchshapeflow.parser import AnnotationParseError, parse_source, parse_tensor_annotation
from torchshapeflow.report import FileReport, HoverFact
from torchshapeflow.rules import (
    infer_binary_broadcast,
    infer_cat,
    infer_chunk,
    infer_conv2d,
    infer_diagonal,
    infer_einsum,
    infer_embedding,
    infer_flatten,
    infer_index_select,
    infer_interpolate,
    infer_linear,
    infer_matmul,
    infer_mm,
    infer_movedim,
    infer_one_hot,
    infer_permute,
    infer_pool2d,
    infer_reduction,
    infer_reshape,
    infer_size,
    infer_split,
    infer_squeeze,
    infer_stack,
    infer_subscript,
    infer_topk,
    infer_transpose,
    infer_unsqueeze,
)
from torchshapeflow.rules.common import int_from_ast, qualified_name

# nn.Module types whose output shape equals their input shape.
_PASSTHROUGH_SUFFIXES: frozenset[str] = frozenset(
    {
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "LayerNorm",
        "Dropout",
        "Dropout2d",
        "Dropout3d",
        "ReLU",
        "LeakyReLU",
        "GELU",
        "SiLU",
        "Sigmoid",
        "Tanh",
        "ELU",
        "SELU",
        "PReLU",
        "Mish",
        "Hardswish",
        "Hardsigmoid",
        "Identity",
        "Softmax",
    }
)

# Reduction ops recognized on both tensors (x.sum) and torch.* functions (torch.sum).
_REDUCTION_OPS: frozenset[str] = frozenset(
    {
        "sum",
        "mean",
        "max",
        "min",
        "amax",
        "amin",
        "prod",
        "all",
        "any",
        "argmax",
        "argmin",
        "nanmean",
        "nansum",
    }
)

# Tensor methods that preserve shape (dtype/device casts, memory management, etc.)
_PASSTHROUGH_METHODS: frozenset[str] = frozenset(
    {
        "contiguous",
        "float",
        "half",
        "double",
        "int",
        "long",
        "short",
        "byte",
        "bool",
        "to",
        "detach",
        "clone",
        "cpu",
        "cuda",
        "type",
        "masked_fill",
        "masked_fill_",
        "requires_grad_",
        "fill_",
        "zero_",
        "normal_",
        "uniform_",
        "flip",
        "abs",
        "neg",
        "sign",
    }
)

# Functional API suffixes whose output shape equals the first argument's shape.
_FUNCTIONAL_PASSTHROUGH: frozenset[str] = frozenset(
    {
        "softmax",
        "log_softmax",
        "relu",
        "relu_",
        "leaky_relu",
        "leaky_relu_",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "elu",
        "selu",
        "mish",
        "hardswish",
        "dropout",
        "dropout2d",
        "dropout3d",
        "layer_norm",
        "batch_norm",
        "group_norm",
        "instance_norm",
        "normalize",
        "triu",
        "tril",
        "flip",
        "isfinite",
        "isinf",
        "isnan",
        "abs",
        "neg",
        "sign",
    }
)

# *_like constructors: output shape equals first argument's shape.
_LIKE_OPS: frozenset[str] = frozenset(
    {"zeros_like", "ones_like", "empty_like", "full_like", "rand_like", "randn_like"}
)

# Size-based constructors: shape is built from positional/keyword size args.
_TENSOR_CONSTRUCTORS: frozenset[str] = frozenset(
    {"zeros", "ones", "empty", "randn", "rand", "full"}
)

# Tensor methods that return non-tensor values (no shape to track).
# These should NOT trigger TSF2001 when they return None from _eval_tensor_method.
_NON_TENSOR_METHODS: frozenset[str] = frozenset({"item", "numpy", "tolist", "dim"})

# Python builtins and common utility functions that should NOT trigger TSF2002.
_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "print",
        "len",
        "range",
        "enumerate",
        "zip",
        "int",
        "float",
        "str",
        "list",
        "tuple",
        "dict",
        "set",
        "type",
        "isinstance",
        "hasattr",
        "getattr",
        "sorted",
        "reversed",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "map",
        "filter",
        "any",
        "all",
        "bool",
        "id",
        "repr",
        "hash",
        "iter",
        "next",
        "super",
        "vars",
        "dir",
        "object",
        "property",
        "staticmethod",
        "classmethod",
    }
)


@dataclass
class ModuleContext:
    path: Path
    diagnostics: list[Diagnostic] = field(default_factory=list)
    hovers: list[HoverFact] = field(default_factory=list)
    aliases: dict[str, TensorValue] = field(default_factory=dict)
    func_sigs: dict[str, FuncSig] = field(default_factory=dict)
    return_shape: TensorValue | None = None
    collected_returns: list[TensorValue | None] = field(default_factory=list)
    in_annotated_function: bool = False

    def error(
        self,
        node: ast.AST,
        code: str,
        message: str,
        severity: Severity = "error",
    ) -> None:
        """Append a diagnostic at the location of *node*.

        Line numbers are 1-based (from ast); column offsets are converted from
        0-based (ast) to 1-based by adding 1.
        """
        self.diagnostics.append(
            Diagnostic(
                code=code,
                message=message,
                path=self.path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                severity=severity,
            )
        )

    def hover(self, name: str, node: ast.AST, tensor: TensorValue) -> None:
        self.hovers.append(
            HoverFact(
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                end_column=getattr(node, "end_col_offset", getattr(node, "col_offset", 0)) + 1,
                name=name,
                shape=str(tensor.shape),
            )
        )


def analyze_path(path: Path, project_index: ProjectIndex | None = None) -> FileReport:
    source = path.read_text(encoding="utf-8")
    return analyze_source(source, path, project_index)


def analyze_source(
    source: str,
    path: Path,
    project_index: ProjectIndex | None = None,
) -> FileReport:
    module = parse_source(source, str(path))

    raw_aliases = collect_raw_aliases(module)
    import_map = collect_imports(module)

    imported_aliases: dict[str, TensorValue] = {}
    imported_funcs: dict[str, FuncSig] = {}
    if project_index is not None:
        for local_name, (module_name, original_name) in import_map.items():
            src_path = project_index.resolve_import(module_name, path)
            if src_path is not None:
                file_data = project_index.index_file(src_path)
                if original_name in file_data.aliases:
                    imported_aliases[local_name] = file_data.aliases[original_name]
                if original_name in file_data.func_sigs:
                    imported_funcs[local_name] = file_data.func_sigs[original_name]

    all_aliases = resolve_aliases(raw_aliases, imported_aliases)

    local_func_sigs: dict[str, FuncSig] = {}
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            sig = extract_func_sig(node, all_aliases)
            if sig is not None:
                local_func_sigs[node.name] = sig

    all_func_sigs = {**imported_funcs, **local_func_sigs}

    context = ModuleContext(path=path, aliases=all_aliases, func_sigs=all_func_sigs)
    class_specs = _collect_class_specs(module)
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            _analyze_function(node, context, {})
        elif isinstance(node, ast.ClassDef):
            specs = class_specs.get(node.name, {})
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "forward":
                    _analyze_function(child, context, specs)
    return FileReport(path=str(path), diagnostics=context.diagnostics, hovers=context.hovers)


def _collect_class_specs(
    module: ast.Module,
) -> dict[str, dict[str, ModuleSpec]]:
    specs: dict[str, dict[str, ModuleSpec]] = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_specs: dict[str, ModuleSpec] = {}
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                init_params = _collect_init_param_defaults(child)
                for statement in child.body:
                    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                        target = statement.targets[0]
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            spec = _parse_module_spec(statement.value, init_params)
                            if spec is not None:
                                class_specs[target.attr] = spec
        if class_specs:
            specs[node.name] = class_specs
    return specs


def _collect_init_param_defaults(func: ast.FunctionDef) -> dict[str, int]:
    """Extract integer default values from ``__init__`` parameters.

    Python aligns defaults right-to-left with parameters, so ``def f(a, b=1, c=2)``
    has ``args.args = [a, b, c]`` and ``args.defaults = [1, 2]``.
    """
    params: dict[str, int] = {}
    args = func.args
    n_defaults = len(args.defaults)
    n_args = len(args.args)
    for i, default in enumerate(args.defaults):
        arg = args.args[n_args - n_defaults + i]
        val = int_from_ast(default)
        if val is not None:
            params[arg.arg] = val
    return params


def _int_from_ast_with_params(node: ast.AST, init_params: dict[str, int] | None) -> int | None:
    """Try ``int_from_ast`` first, then fall back to ``__init__`` parameter defaults."""
    val = int_from_ast(node)
    if val is not None:
        return val
    if init_params and isinstance(node, ast.Name) and node.id in init_params:
        return init_params[node.id]
    return None


def _parse_module_spec(
    node: ast.AST,
    init_params: dict[str, int] | None = None,
) -> ModuleSpec | None:
    if not isinstance(node, ast.Call):
        return None
    name = qualified_name(node.func)
    short_name = name.split(".")[-1]

    def _int(n: ast.AST) -> int | None:
        return _int_from_ast_with_params(n, init_params)

    if name.endswith("Linear") and len(node.args) >= 2:
        in_features = _int(node.args[0])  # may be None (non-literal)
        out_features = _int(node.args[1])
        if out_features is not None:
            return LinearSpec(in_features=in_features, out_features=out_features)
    if name.endswith("Conv2d") and len(node.args) >= 3:
        in_channels = _int(node.args[0])  # may be None (non-literal)
        out_channels = _int(node.args[1])
        kernel_size = _int_pair(node.args[2])
        stride = _int_pair(_keyword_or_default(node, "stride"), default=(1, 1))
        padding = _int_pair(_keyword_or_default(node, "padding"), default=(0, 0))
        dilation = _int_pair(_keyword_or_default(node, "dilation"), default=(1, 1))
        if None not in (out_channels, kernel_size, stride, padding, dilation):
            assert out_channels is not None
            assert kernel_size is not None
            assert stride is not None
            assert padding is not None
            assert dilation is not None
            return Conv2dSpec(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
    if short_name == "Embedding" and len(node.args) >= 2:
        embedding_dim = _int(node.args[1])
        if embedding_dim is not None:
            return EmbeddingSpec(embedding_dim=embedding_dim)
    if name.endswith("MaxPool2d") and node.args:
        kernel_size = _int_pair(node.args[0])
        if kernel_size is not None:
            stride = _pool_stride(node, kernel_size)
            padding = _int_pair(_keyword_or_default(node, "padding"), default=(0, 0))
            dilation = _int_pair(_keyword_or_default(node, "dilation"), default=(1, 1))
            if stride is not None and padding is not None and dilation is not None:
                return Pool2dSpec(
                    kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
                )
    if name.endswith("AvgPool2d") and node.args:
        kernel_size = _int_pair(node.args[0])
        if kernel_size is not None:
            stride = _pool_stride(node, kernel_size)
            padding = _int_pair(_keyword_or_default(node, "padding"), default=(0, 0))
            if stride is not None and padding is not None:
                return Pool2dSpec(
                    kernel_size=kernel_size, stride=stride, padding=padding, dilation=(1, 1)
                )
    if short_name in _PASSTHROUGH_SUFFIXES:
        return PassthroughSpec()
    if short_name == "Sequential":
        sub_specs: list[ModuleSpec] = []
        for arg in node.args:
            sub = _parse_module_spec(arg, init_params)
            if sub is not None:
                sub_specs.append(sub)
        return SequentialSpec(specs=tuple(sub_specs))
    if short_name == "MultiheadAttention" and len(node.args) >= 2:
        embed_dim = _int(node.args[0])
        num_heads = _int(node.args[1])
        batch_first_node = _keyword_or_default(node, "batch_first")
        batch_first = False
        if isinstance(batch_first_node, ast.Constant) and isinstance(batch_first_node.value, bool):
            batch_first = batch_first_node.value
        if embed_dim is not None and num_heads is not None:
            return MultiheadAttentionSpec(
                embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first
            )
    return None


def _keyword_or_default(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _int_pair(
    node: ast.AST | None,
    default: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    if node is None:
        return default
    single = int_from_ast(node)
    if single is not None:
        return single, single
    if isinstance(node, ast.Tuple) and len(node.elts) == 2:
        first = int_from_ast(node.elts[0])
        second = int_from_ast(node.elts[1])
        if first is not None and second is not None:
            return first, second
    return None


def _pool_stride(call: ast.Call, kernel_size: tuple[int, int]) -> tuple[int, int] | None:
    """Return the effective stride for a pooling call.

    PyTorch pool layers default stride to kernel_size when the argument is absent.
    Checks positional arg[1] first, then the ``stride`` keyword.
    """
    stride_node: ast.AST | None = None
    if len(call.args) >= 2:
        stride_node = call.args[1]
    else:
        stride_node = _keyword_or_default(call, "stride")
    if stride_node is None:
        return kernel_size
    return _int_pair(stride_node)


def _analyze_function(
    function: ast.FunctionDef,
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> None:
    env: dict[str, Value] = {}
    tensor_params: list[tuple[str, TensorValue]] = []
    for argument in function.args.args:
        if argument.arg == "self":
            continue
        if argument.annotation is None:
            continue
        try:
            tensor = parse_tensor_annotation(argument.annotation, context.aliases)
        except AnnotationParseError as error:
            context.error(argument, "TSF1001", error.message)
            continue
        if tensor is not None:
            env[argument.arg] = tensor
            context.hover(argument.arg, argument, tensor)
            tensor_params.append((argument.arg, tensor))
    # Track whether this function has tensor-annotated parameters (for TSF2xxx warnings).
    old_in_annotated = context.in_annotated_function
    context.in_annotated_function = len(tensor_params) > 0
    # Parse return annotation; set on context so _analyze_statement can validate.
    old_return_shape = context.return_shape
    old_collected_returns = context.collected_returns
    context.collected_returns = []
    if function.returns is not None:
        try:
            context.return_shape = parse_tensor_annotation(function.returns, context.aliases)
        except AnnotationParseError:
            context.return_shape = None
    else:
        context.return_shape = None
    for statement in function.body:
        _analyze_statement(statement, env, context, module_specs)
    # Emit a signature hover on the function name if any tensor params are present.
    if tensor_params:
        _emit_signature_hover(
            function, tensor_params, context.return_shape, context.collected_returns, context
        )
    context.collected_returns = old_collected_returns
    context.return_shape = old_return_shape
    context.in_annotated_function = old_in_annotated


def _emit_signature_hover(
    function: ast.FunctionDef,
    tensor_params: list[tuple[str, TensorValue]],
    declared_return: TensorValue | None,
    collected_returns: list[TensorValue | None],
    context: ModuleContext,
) -> None:
    # Format parameter block — one per line when there are multiple.
    if len(tensor_params) == 1:
        params_str = f"({tensor_params[0][0]}: {tensor_params[0][1].shape})"
    else:
        inner = ",\n".join(f"  {name}: {tv.shape}" for name, tv in tensor_params)
        params_str = f"(\n{inner}\n)"

    # Determine return string — prefer inferred over declared annotation.
    tensor_returns = [tv for tv in collected_returns if tv is not None]
    if tensor_returns:
        unique = list(dict.fromkeys(str(tv.shape) for tv in tensor_returns))
        if len(unique) == 1:
            return_str = f" → {unique[0]}"
        else:
            cases = "\n".join(
                f"  - {s}" if tv is not None else "  - ?"
                for tv, s in zip(
                    collected_returns,
                    (str(tv.shape) if tv is not None else "?" for tv in collected_returns),
                    strict=False,
                )
            )
            return_str = f" →\n{cases}"
    elif declared_return is not None:
        return_str = f" → {declared_return.shape}"
    else:
        return_str = ""

    sig = params_str + return_str
    # Emit the hover at the function name token (after "def ").
    name_col = function.col_offset + 4  # 0-based start of the name
    name_end_col = name_col + len(function.name)
    context.hovers.append(
        HoverFact(
            line=function.lineno,
            column=name_col + 1,  # 1-based
            end_line=function.lineno,
            end_column=name_end_col + 1,  # 1-based
            name=function.name,
            shape=sig,
        )
    )


def _analyze_statement(
    statement: ast.stmt,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> None:
    if isinstance(statement, ast.Assign):
        value = _eval_expr(statement.value, env, context, module_specs)
        for target in statement.targets:
            if isinstance(target, ast.Tuple) and isinstance(value, TensorTupleValue):
                for t_elt, tv in zip(target.elts, value.tensors, strict=False):
                    _bind_target(t_elt, tv, env, context)
            else:
                _bind_target(target, value, env, context)
        return
    if isinstance(statement, ast.AugAssign):
        # x += y  →  treat as x = x <op> y, updating env with the broadcast result.
        target_name = statement.target.id if isinstance(statement.target, ast.Name) else None
        lhs = env.get(target_name) if target_name else None
        rhs = _eval_expr(statement.value, env, context, module_specs)
        if (
            isinstance(lhs, TensorValue)
            and isinstance(rhs, TensorValue)
            and isinstance(
                statement.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv, ast.Pow)
            )
        ):
            result = infer_binary_broadcast(lhs, rhs)
            if result is not None and target_name is not None:
                env[target_name] = result
                context.hover(target_name, statement.target, result)
        return
    if isinstance(statement, ast.AnnAssign):
        value = (
            _eval_expr(statement.value, env, context, module_specs)
            if statement.value is not None
            else None
        )
        _bind_target(statement.target, value, env, context)
        return
    if isinstance(statement, ast.Return) and statement.value is not None:
        actual = _eval_expr(statement.value, env, context, module_specs)
        context.collected_returns.append(actual if isinstance(actual, TensorValue) else None)
        if isinstance(actual, TensorValue):
            if not isinstance(statement.value, ast.Name):
                context.hover("<return>", statement.value, actual)
            if context.return_shape is not None and _shapes_definitely_mismatch(
                context.return_shape.shape, actual.shape
            ):
                context.error(
                    statement,
                    "TSF1009",
                    f"Return shape {actual.shape} does not match declared"
                    f" {context.return_shape.shape}.",
                )
        return
    if isinstance(statement, ast.Expr):
        _eval_expr(statement.value, env, context, module_specs)
        return
    if isinstance(statement, ast.If):
        _analyze_if(statement, env, context, module_specs)
        return


def _analyze_if(
    node: ast.If,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> None:
    """Analyze an if/else block by walking both branches and merging environments.

    If both branches exist, variables assigned with the same shape in both are
    kept; variables assigned with different shapes get ``UnknownDim("?")`` for
    differing dimensions. Variables assigned in only one branch are dropped.
    If there is no ``else``, assignments from the ``if`` body are kept (since
    the condition may or may not hold, keeping them is pragmatically useful
    for patterns like ``if mask is not None: mask = mask.unsqueeze(1)``).
    """
    pre_env = dict(env)
    env_then: dict[str, Value] = dict(env)
    for stmt in node.body:
        _analyze_statement(stmt, env_then, context, module_specs)
    if node.orelse:
        env_else: dict[str, Value] = dict(pre_env)
        for stmt in node.orelse:
            _analyze_statement(stmt, env_else, context, module_specs)
        _merge_envs(env, pre_env, env_then, env_else)
    else:
        # No else: take the ``if`` body environment (pragmatically useful).
        env.update(env_then)


def _merge_envs(
    env: dict[str, Value],
    pre_env: dict[str, Value],
    env_then: dict[str, Value],
    env_else: dict[str, Value],
) -> None:
    """Merge two branch environments into *env*.

    For each key in the union of ``env_then`` and ``env_else``:
    - If both have the same shape (by string) → keep it.
    - If both are TensorValues with the same rank but different dims → merge
      dimension-wise, keeping matching dims and using ``UnknownDim("?")`` for
      differing ones.
    - If the key existed before the ``if`` and was not changed by either → keep it.
    - Otherwise → drop the key from env.

    Invariant: both branch envs must be initialized as copies of ``pre_env``
    so that pre-existing variables appear in both and are preserved.
    """
    all_keys = set(env_then) | set(env_else)
    env.clear()
    for key in all_keys:
        then_val = env_then.get(key)
        else_val = env_else.get(key)
        if then_val is not None and else_val is not None:
            if isinstance(then_val, TensorValue) and isinstance(else_val, TensorValue):
                if str(then_val.shape) == str(else_val.shape):
                    env[key] = then_val
                elif then_val.rank == else_val.rank:
                    merged_dims: list[Dim] = []
                    for d1, d2 in zip(then_val.shape.dims, else_val.shape.dims, strict=True):
                        if render_dim(d1) == render_dim(d2):
                            merged_dims.append(d1)
                        else:
                            merged_dims.append(UnknownDim("?"))
                    env[key] = TensorValue(TensorShape(tuple(merged_dims)))
                # Different ranks: drop the key
            elif type(then_val) is type(else_val) and then_val == else_val:
                env[key] = then_val
            # Different types or values: drop the key
        elif key in pre_env:
            env[key] = pre_env[key]


def _bind_target(
    target: ast.AST,
    value: Value | Dim | None,
    env: dict[str, Value],
    context: ModuleContext,
) -> None:
    if not isinstance(target, ast.Name):
        return
    if isinstance(
        value,
        (
            TensorValue,
            ShapeTupleValue,
            IntegerValue,
            LinearSpec,
            Conv2dSpec,
            PassthroughSpec,
            EmbeddingSpec,
            Pool2dSpec,
            MultiheadAttentionSpec,
            SequentialSpec,
            TensorTupleValue,
        ),
    ):
        env[target.id] = value
        if isinstance(value, TensorValue):
            context.hover(target.id, target, value)


def _eval_expr(
    node: ast.AST,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> Value | Dim | None:
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        if isinstance(value, TensorValue):
            context.hover(node.id, node, value)
        return value
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return IntegerValue(node.value)
    if isinstance(node, ast.Attribute):
        base = _eval_expr(node.value, env, context, module_specs)
        if isinstance(base, TensorValue) and node.attr == "shape":
            return ShapeTupleValue(base.shape.dims)
        if isinstance(base, TensorValue) and node.attr == "ndim":
            return IntegerValue(base.rank)
        if isinstance(base, TensorValue) and node.attr in {"values", "indices"}:
            return base
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            return module_specs.get(node.attr)
        return None
    if isinstance(node, ast.Subscript):
        base = _eval_expr(node.value, env, context, module_specs)
        if isinstance(base, (TensorValue, ShapeTupleValue)):
            return infer_subscript(base, node)
        if isinstance(base, TensorTupleValue):
            idx = int_from_ast(node.slice)
            if idx is not None:
                norm = normalize_index(idx, len(base.tensors))
                if norm is not None:
                    return base.tensors[norm]
        return None
    _element_wise_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _element_wise_ops):
        left = _eval_expr(node.left, env, context, module_specs)
        right = _eval_expr(node.right, env, context, module_specs)
        if isinstance(left, TensorValue) and isinstance(right, TensorValue):
            result = infer_binary_broadcast(left, right)
            if result is None:
                context.error(node, "TSF1006", "Broadcasting incompatibility.")
            elif broadcast_has_uncertain_dims(left.shape, right.shape):
                context.error(
                    node,
                    "TSF1006",
                    "Broadcasting compatibility cannot be verified statically"
                    f" ({left.shape} vs {right.shape}).",
                    severity="warning",
                )
            return result
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        left = _eval_expr(node.left, env, context, module_specs)
        right = _eval_expr(node.right, env, context, module_specs)
        if isinstance(left, TensorValue) and isinstance(right, TensorValue):
            result = infer_matmul(left, right)
            if result is None:
                context.error(node, "TSF1003", "Incompatible matmul shapes.")
            return result
        return None
    if isinstance(node, ast.Call):
        return _eval_call(node, env, context, module_specs)
    if isinstance(node, ast.Tuple):
        return None
    return None


def _apply_module_spec(
    spec: ModuleSpec,
    tensor: TensorValue,
    node: ast.AST,
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> TensorValue | TensorTupleValue | None:
    """Apply a single module spec to an input tensor and return the output."""
    if isinstance(spec, LinearSpec):
        result = infer_linear(spec, tensor)
        if result is None:
            context.error(node, "TSF1007", "nn.Linear input shape mismatch.")
        return result
    if isinstance(spec, Conv2dSpec):
        result = infer_conv2d(spec, tensor)
        if result is None:
            context.error(node, "TSF1007", "nn.Conv2d input shape mismatch.")
        return result
    if isinstance(spec, PassthroughSpec):
        return tensor
    if isinstance(spec, EmbeddingSpec):
        return infer_embedding(spec, tensor)
    if isinstance(spec, Pool2dSpec):
        result = infer_pool2d(spec, tensor)
        if result is None:
            context.error(node, "TSF1007", "nn.MaxPool2d/AvgPool2d requires rank-4 input.")
        return result
    if isinstance(spec, SequentialSpec):
        current = tensor
        for sub in spec.specs:
            out = _apply_module_spec(sub, current, node, context, module_specs)
            if not isinstance(out, TensorValue):
                return None
            current = out
        return current
    if isinstance(spec, MultiheadAttentionSpec):
        # Output has same shape as query; weights shape is not statically known.
        output = TensorValue(tensor.shape)
        weights = TensorValue(TensorShape((UnknownDim("?"), UnknownDim("?"), UnknownDim("?"))))
        return TensorTupleValue((output, weights))
    return None


def _eval_call(
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> Value | None:
    callee_name = qualified_name(node.func)
    if isinstance(node.func, ast.Attribute):
        owner: Value | Dim | None
        is_self_call = isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
        if is_self_call:
            owner = module_specs.get(node.func.attr)
        else:
            owner = _eval_expr(node.func.value, env, context, module_specs)
        if isinstance(owner, TensorValue):
            return _eval_tensor_method(owner, node, context, env, module_specs)
        if isinstance(
            owner,
            (
                LinearSpec,
                Conv2dSpec,
                PassthroughSpec,
                EmbeddingSpec,
                Pool2dSpec,
                SequentialSpec,
                MultiheadAttentionSpec,
            ),
        ):
            argument = _eval_expr(node.args[0], env, context, module_specs) if node.args else None
            if isinstance(argument, TensorValue):
                return _apply_module_spec(owner, argument, node, context, module_specs)
        # TSF2003: self.xxx(tensor) where xxx has no spec.
        if is_self_call and owner is None and context.in_annotated_function:
            has_tensor_arg = False
            for arg_node in node.args:
                arg_val = _eval_expr(arg_node, env, context, module_specs)
                if isinstance(arg_val, TensorValue):
                    has_tensor_arg = True
                    break
            if has_tensor_arg:
                attr = node.func.attr
                context.error(
                    node,
                    "TSF2003",
                    f"No shape spec for 'self.{attr}' — shape not tracked.",
                    severity="warning",
                )
    if callee_name.endswith("reshape") and len(node.args) >= 2:
        tensor = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor, TensorValue):
            return _reshape_from_args(tensor, node.args[1:], context, node, env, module_specs)
    if callee_name.endswith("cat") and node.args:
        values = _tensor_sequence(node.args[0], env, context, module_specs)
        dim = _keyword_int(node, "dim", 0)
        if values is not None and dim is not None:
            result = infer_cat(values, dim)
            if result is None:
                context.error(
                    node,
                    "TSF1005",
                    "Invalid concat dimension or mismatched input shapes.",
                )
            return result
    if callee_name.endswith("stack") and node.args:
        values = _tensor_sequence(node.args[0], env, context, module_specs)
        dim = _keyword_int(node, "dim", 0)
        if values is not None and dim is not None:
            result = infer_stack(values, dim)
            if result is None:
                context.error(
                    node,
                    "TSF1005",
                    "Invalid stack dimension or mismatched input shapes.",
                )
            return result
    if callee_name.endswith("matmul") or callee_name.endswith("bmm"):
        if len(node.args) >= 2:
            left = _eval_expr(node.args[0], env, context, module_specs)
            right = _eval_expr(node.args[1], env, context, module_specs)
            if isinstance(left, TensorValue) and isinstance(right, TensorValue):
                result = infer_matmul(left, right)
                if result is None:
                    context.error(node, "TSF1003", "Incompatible matmul shapes.")
                return result
    if callee_name.endswith(".mm") or callee_name == "mm":
        if len(node.args) >= 2:
            left = _eval_expr(node.args[0], env, context, module_specs)
            right = _eval_expr(node.args[1], env, context, module_specs)
            if isinstance(left, TensorValue) and isinstance(right, TensorValue):
                result = infer_mm(left, right)
                if result is None:
                    context.error(node, "TSF1003", "Incompatible mm shapes.")
                return result
    if callee_name.endswith(".movedim") or callee_name == "movedim":
        if len(node.args) >= 3:
            tensor = _eval_expr(node.args[0], env, context, module_specs)
            if isinstance(tensor, TensorValue):
                src = _int_or_tuple(node.args[1])
                dst = _int_or_tuple(node.args[2])
                if src is not None and dst is not None:
                    return infer_movedim(tensor, src, dst)
    # torch.einsum(subscript, t1, t2, ...) or torch.einsum(subscript, [t1, t2]).
    if callee_name.endswith("einsum") and node.args:
        subscript_node = node.args[0]
        if isinstance(subscript_node, ast.Constant) and isinstance(subscript_node.value, str):
            subscript_str = subscript_node.value
            if len(node.args) == 2 and isinstance(node.args[1], (ast.List, ast.Tuple)):
                tensor_arg_nodes: list[ast.expr] = list(node.args[1].elts)
            else:
                tensor_arg_nodes = list(node.args[1:])
            tensor_vals = [_eval_expr(a, env, context, module_specs) for a in tensor_arg_nodes]
            tensor_list = [t for t in tensor_vals if isinstance(t, TensorValue)]
            if len(tensor_list) == len(tensor_arg_nodes) and "->" in subscript_str:
                result = infer_einsum(subscript_str, tensor_list)
                if result is None:
                    context.error(node, "TSF1003", "Incompatible einsum shapes.")
                return result
    # F.interpolate(x, size=(H, W)) / F.interpolate(x, scale_factor=2.0).
    if callee_name.endswith("interpolate") and node.args:
        tensor = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor, TensorValue):
            n_spatial = tensor.rank - 2
            if n_spatial > 0:
                size_dims = _interpolate_size_arg(node, n_spatial, env, context, module_specs)
                scale = _interpolate_scale_arg(node, n_spatial)
                if size_dims is not None or scale is not None:
                    result = infer_interpolate(tensor, size_dims, scale)
                    if result is not None:
                        return result
    # torch.sum(x, dim=1) / torch.mean(x) / etc. — global reduction functions.
    callee_leaf = callee_name.split(".")[-1]
    if callee_leaf in _REDUCTION_OPS and node.args:
        tensor = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor, TensorValue):
            rdim = _reduction_dim(node, arg_offset=1)
            keepdim = _reduction_keepdim(node, positional_index=2)
            return infer_reduction(tensor, rdim, keepdim)
    # Functional passthrough: torch.softmax(x, dim=-1), F.relu(x), torch.triu(x), etc.
    if callee_leaf in _FUNCTIONAL_PASSTHROUGH and node.args:
        first_arg = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(first_arg, TensorValue):
            return first_arg
    # F.one_hot(x, num_classes=N).
    if callee_leaf == "one_hot" and node.args:
        tensor = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor, TensorValue):
            num_classes: int | None = None
            if len(node.args) >= 2:
                num_classes = int_from_ast(node.args[1])
            if num_classes is None:
                nc_node = _keyword_or_default(node, "num_classes")
                if nc_node is not None:
                    num_classes = int_from_ast(nc_node)
            if num_classes is not None:
                return infer_one_hot(tensor, num_classes)
            return TensorValue(TensorShape(tensor.shape.dims + (UnknownDim("?"),)))
    # torch.topk(x, k, dim).
    if callee_leaf == "topk" and node.args:
        tensor = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor, TensorValue):
            k_val = int_from_ast(node.args[1]) if len(node.args) >= 2 else None
            if k_val is None:
                k_node = _keyword_or_default(node, "k")
                if k_node is not None:
                    k_val = int_from_ast(k_node)
            dim_val = _positional_int(node.args, 2, -1)
            if dim_val is None:
                dim_val = _keyword_int(node, "dim", -1)
            if dim_val is None:
                dim_val = -1
            if k_val is not None:
                result = infer_topk(tensor, k_val, dim_val)
                if result is not None:
                    return result
    # torch.bincount — runtime-dependent 1-D output.
    if callee_leaf == "bincount" and node.args:
        return TensorValue(TensorShape((UnknownDim("?"),)))
    # torch.diagonal(x, offset, dim1, dim2) — functional form.
    if callee_leaf == "diagonal" and node.args:
        tensor = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor, TensorValue):
            offset_val = _positional_int(node.args, 1, None)
            if offset_val is None:
                offset_val = _keyword_int(node, "offset", 0)
            if offset_val is None:
                offset_val = 0
            dim1_val = _positional_int(node.args, 2, None)
            if dim1_val is None:
                dim1_val = _keyword_int(node, "dim1", 0)
            if dim1_val is None:
                dim1_val = 0
            dim2_val = _positional_int(node.args, 3, None)
            if dim2_val is None:
                dim2_val = _keyword_int(node, "dim2", 1)
            if dim2_val is None:
                dim2_val = 1
            return infer_diagonal(tensor, offset_val, dim1_val, dim2_val)
    # *_like constructors: torch.zeros_like(x), etc.
    if callee_leaf in _LIKE_OPS and node.args:
        first_arg = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(first_arg, TensorValue):
            return TensorValue(first_arg.shape)
    # Size-based constructors: torch.zeros(B, T, D), torch.ones((B, T)), etc.
    if callee_leaf in _TENSOR_CONSTRUCTORS:
        dims = _constructor_size(node, callee_leaf, env, context, module_specs)
        if dims is not None:
            return TensorValue(TensorShape(tuple(dims)))
    # torch.arange(n) → 1-D tensor.
    if callee_leaf == "arange" and node.args:
        arange_len = _arange_length(node)
        if arange_len is not None:
            return TensorValue(TensorShape((ConstantDim(arange_len),)))
        arange_dim: Dim = (
            _size_to_dim(node.args[0], env, context, module_specs)
            if len(node.args) == 1
            else UnknownDim("?")
        )
        return TensorValue(TensorShape((arange_dim,)))
    # F.scaled_dot_product_attention(q, k, v, ...) → same shape as q.
    if callee_name.endswith("scaled_dot_product_attention") and node.args:
        q_val = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(q_val, TensorValue):
            return q_val
    # torch.split(tensor, split_size_or_sections, dim=0)
    if callee_leaf == "split" and len(node.args) >= 2:
        tensor_arg = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor_arg, TensorValue):
            size_node = node.args[1]
            dim = _positional_int(node.args, 2, None)
            if dim is None:
                dim = _keyword_int(node, "dim", 0)
            if dim is None:
                dim = 0
            if isinstance(size_node, (ast.List, ast.Tuple)):
                sizes = [int_from_ast(e) for e in size_node.elts]
                if all(s is not None for s in sizes):
                    split_result = infer_split(tensor_arg, [s for s in sizes if s is not None], dim)
                    if split_result is not None:
                        return split_result
            else:
                split_size = int_from_ast(size_node)
                if split_size is not None:
                    split_result = infer_split(tensor_arg, split_size, dim)
                    if split_result is not None:
                        return split_result
    # Module alias: m = self.linear; m(x) — look up spec stored in env.
    if isinstance(node.func, ast.Name):
        env_val = env.get(node.func.id)
        if isinstance(
            env_val,
            (
                LinearSpec,
                Conv2dSpec,
                PassthroughSpec,
                EmbeddingSpec,
                Pool2dSpec,
                SequentialSpec,
                MultiheadAttentionSpec,
            ),
        ):
            argument = _eval_expr(node.args[0], env, context, module_specs) if node.args else None
            if isinstance(argument, TensorValue):
                return _apply_module_spec(env_val, argument, node, context, module_specs)
    # User-defined / cross-file function call: look up in func_sigs.
    if isinstance(node.func, ast.Name):
        sig = context.func_sigs.get(node.func.id)
        if sig is not None and sig.return_shape is not None:
            mapping: dict[str, Dim] = {}
            for param_tv, arg_node in zip(sig.param_shapes, node.args, strict=False):
                if param_tv is None:
                    continue
                arg_val = _eval_expr(arg_node, env, context, module_specs)
                if isinstance(arg_val, TensorValue):
                    sub = unify_dims(param_tv.shape.dims, arg_val.shape.dims)
                    for sym_name, bound_dim in sub.items():
                        if sym_name in mapping and render_dim(mapping[sym_name]) != render_dim(
                            bound_dim
                        ):
                            context.error(
                                node,
                                "TSF1010",
                                f"Symbolic dim '{sym_name}' bound to conflicting values: "
                                f"{render_dim(mapping[sym_name])} vs {render_dim(bound_dim)}.",
                            )
                            mapping[sym_name] = UnknownDim("?")
                        else:
                            mapping[sym_name] = bound_dim
            return TensorValue(apply_substitution(sig.return_shape.shape, mapping))
    # TSF2002: call to unannotated function with tensor arg in annotated function.
    if (
        isinstance(node.func, ast.Name)
        and context.in_annotated_function
        and node.func.id not in _BUILTIN_NAMES
        and node.func.id not in context.func_sigs
    ):
        # Check that it is not a known module spec alias already handled above.
        env_val = env.get(node.func.id)
        is_module_spec = isinstance(
            env_val,
            (
                LinearSpec,
                Conv2dSpec,
                PassthroughSpec,
                EmbeddingSpec,
                Pool2dSpec,
                SequentialSpec,
                MultiheadAttentionSpec,
            ),
        )
        if not is_module_spec:
            has_tensor_arg = False
            for arg_node in node.args:
                arg_val = _eval_expr(arg_node, env, context, module_specs)
                if isinstance(arg_val, TensorValue):
                    has_tensor_arg = True
                    break
            if has_tensor_arg:
                func_name = node.func.id
                context.error(
                    node,
                    "TSF2002",
                    f"Call to unannotated function '{func_name}'"
                    " — shape not tracked. Consider adding a Shape annotation.",
                    severity="warning",
                )
    return None


def _eval_tensor_method(
    tensor: TensorValue,
    node: ast.Call,
    context: ModuleContext,
    env: dict[str, Value],
    module_specs: dict[str, ModuleSpec],
) -> Value | None:
    assert isinstance(node.func, ast.Attribute)
    name = node.func.attr
    if name in {"reshape", "view"}:
        result = _reshape_from_args(tensor, node.args, context, node, env, module_specs)
        if result is None:
            context.error(node, "TSF1004", "Invalid reshape.")
        return result
    if name == "permute":
        order = tuple(int_from_ast(arg) for arg in node.args)
        if any(item is None for item in order):
            return None
        result = infer_permute(tensor, tuple(item for item in order if item is not None))
        if result is None:
            context.error(node, "TSF1008", "Invalid permutation.")
        return result
    if name == "transpose" and len(node.args) == 2:
        first = int_from_ast(node.args[0])
        second = int_from_ast(node.args[1])
        if first is None or second is None:
            return None
        result = infer_transpose(tensor, first, second)
        if result is None:
            context.error(node, "TSF1008", "Invalid transpose dimensions.")
        return result
    if name == "flatten":
        start_dim = _positional_int(node.args, 0, 0)
        end_dim = _positional_int(node.args, 1, -1)
        if start_dim is None or end_dim is None:
            return None
        result = infer_flatten(tensor, start_dim, end_dim)
        if result is None:
            context.error(node, "TSF1004", "Invalid flatten dimensions.")
        return result
    if name == "squeeze":
        dim = _positional_int(node.args, 0, None)
        result = infer_squeeze(tensor, dim)
        if result is None:
            context.error(node, "TSF1008", "Invalid squeeze dimension.")
        return result
    if name == "unsqueeze" and node.args:
        dim = int_from_ast(node.args[0])
        if dim is None:
            return None
        result = infer_unsqueeze(tensor, dim)
        if result is None:
            context.error(node, "TSF1008", "Invalid unsqueeze dimension.")
        return result
    if name == "size":
        dim = _positional_int(node.args, 0, None)
        return infer_size(tensor, dim)
    if name == "matmul" and node.args:
        right = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(right, TensorValue):
            result = infer_matmul(tensor, right)
            if result is None:
                context.error(node, "TSF1003", "Incompatible matmul shapes.")
            return result
    if name == "mm" and node.args:
        right = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(right, TensorValue):
            result = infer_mm(tensor, right)
            if result is None:
                context.error(node, "TSF1003", "Incompatible mm shapes.")
            return result
    if name in _REDUCTION_OPS:
        rdim = _reduction_dim(node, arg_offset=0)
        keepdim = _reduction_keepdim(node, positional_index=1)
        return infer_reduction(tensor, rdim, keepdim)
    if name in _PASSTHROUGH_METHODS:
        return tensor
    if name == "expand" and node.args:
        return _infer_expand(tensor, node, env, context, module_specs)
    if name == "expand_as" and node.args:
        other = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(other, TensorValue):
            return TensorValue(other.shape)
    if name == "repeat" and node.args:
        return _infer_repeat(tensor, node, env, context, module_specs)
    if name == "chunk" and node.args:
        n = int_from_ast(node.args[0])
        if n is not None:
            dim = _positional_int(node.args, 1, None)
            if dim is None:
                dim = _keyword_int(node, "dim", 0)
            if dim is not None:
                chunk_result = infer_chunk(tensor, n, dim)
                return chunk_result
    if name == "split" and node.args:
        split_result = _split_from_call(tensor, node)
        if split_result is not None:
            return split_result
    if name == "movedim" and len(node.args) >= 2:
        src = _int_or_tuple(node.args[0])
        dst = _int_or_tuple(node.args[1])
        if src is not None and dst is not None:
            return infer_movedim(tensor, src, dst)
    if name == "diagonal":
        offset_val = _positional_int(node.args, 0, 0)
        if offset_val is None:
            offset_val = 0
        dim1_val = _positional_int(node.args, 1, None)
        if dim1_val is None:
            dim1_val = _keyword_int(node, "dim1", 0)
        if dim1_val is None:
            dim1_val = 0
        dim2_val = _positional_int(node.args, 2, None)
        if dim2_val is None:
            dim2_val = _keyword_int(node, "dim2", 1)
        if dim2_val is None:
            dim2_val = 1
        return infer_diagonal(tensor, offset_val, dim1_val, dim2_val)
    if name == "index_select" and len(node.args) >= 2:
        dim_val = int_from_ast(node.args[0])
        if dim_val is not None:
            idx_val = _eval_expr(node.args[1], env, context, module_specs)
            if isinstance(idx_val, TensorValue) and idx_val.rank == 1:
                index_len: Dim = idx_val.shape.dims[0]
            else:
                index_len = UnknownDim("?")
            return infer_index_select(tensor, dim_val, index_len)
    if name == "topk" and node.args:
        k_val = int_from_ast(node.args[0])
        if k_val is None:
            k_node = _keyword_or_default(node, "k")
            if k_node is not None:
                k_val = int_from_ast(k_node)
        dim_val = _positional_int(node.args, 1, None)
        if dim_val is None:
            dim_val = _keyword_int(node, "dim", -1)
        if dim_val is None:
            dim_val = -1
        if k_val is not None:
            result = infer_topk(tensor, k_val, dim_val)
            if result is not None:
                return result
    if name == "numel":
        return IntegerValue(None)
    if name in _NON_TENSOR_METHODS:
        return None
    # TSF2001: warn about unsupported tensor method in annotated function.
    if context.in_annotated_function:
        context.error(
            node,
            "TSF2001",
            f"Unsupported tensor method '.{name}' — shape not tracked.",
            severity="warning",
        )
    return None


def _reshape_from_args(
    tensor: TensorValue,
    args: Sequence[ast.expr],
    context: ModuleContext,
    node: ast.AST,
    env: dict[str, Value],
    module_specs: dict[str, ModuleSpec],
) -> TensorValue | None:
    requested: list[Dim | int] = []
    if len(args) == 1 and isinstance(args[0], ast.Tuple):
        flattened_args = list(args[0].elts)
    else:
        flattened_args = list(args)
    for arg in flattened_args:
        requested_dim = _dim_from_expr(arg, env, context, module_specs)
        if requested_dim is None:
            context.error(node, "TSF1004", "Unsupported reshape dimension expression.")
            return None
        requested.append(requested_dim)
    return infer_reshape(tensor, tuple(requested))


def _dim_from_expr(
    node: ast.AST,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> Dim | int | None:
    integer = int_from_ast(node)
    if integer is not None:
        return integer
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return make_dim(node.value)
    value = _eval_expr(node, env, context, module_specs)
    if isinstance(value, IntegerValue):
        if value.value is not None:
            return value.value
        # Unknown integer — use the variable name as a symbolic dim if possible.
        if isinstance(node, ast.Name):
            return SymbolicDim(node.id)
        return UnknownDim("?")
    if isinstance(value, (ConstantDim, ExpressionDim, SymbolicDim, UnknownDim)):
        return value
    # Unresolved name (not in env) — treat as a symbolic dimension.
    if value is None and isinstance(node, ast.Name):
        return SymbolicDim(node.id)
    return None


def _tensor_sequence(
    node: ast.AST,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> tuple[TensorValue, ...] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[TensorValue] = []
    for element in node.elts:
        value = _eval_expr(element, env, context, module_specs)
        if not isinstance(value, TensorValue):
            return None
        values.append(value)
    return tuple(values)


def _int_or_tuple(node: ast.expr) -> int | tuple[int, ...] | None:
    """Parse an AST node as an int or a tuple/list of ints (for movedim source/destination)."""
    v = int_from_ast(node)
    if v is not None:
        return v
    if isinstance(node, (ast.Tuple, ast.List)):
        parts: list[int] = []
        for elt in node.elts:
            i = int_from_ast(elt)
            if i is None:
                return None
            parts.append(i)
        return tuple(parts)
    return None


def _keyword_int(node: ast.Call, name: str, default: int | None) -> int | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return int_from_ast(keyword.value)
    return default


def _positional_int(
    args: Sequence[ast.expr],
    index: int,
    default: int | None,
) -> int | None:
    if index >= len(args):
        return default
    return int_from_ast(args[index])


def _reduction_dim(node: ast.Call, arg_offset: int) -> int | tuple[int, ...] | None:
    """Extract the ``dim`` argument from a reduction call."""
    dim_node: ast.AST | None = None
    if len(node.args) > arg_offset:
        dim_node = node.args[arg_offset]
    else:
        for kw in node.keywords:
            if kw.arg == "dim":
                dim_node = kw.value
                break
    if dim_node is None:
        return None
    if isinstance(dim_node, ast.Tuple):
        ints = [int_from_ast(elt) for elt in dim_node.elts]
        if any(i is None for i in ints):
            return None
        return tuple(int(i) for i in ints if i is not None)
    return int_from_ast(dim_node)


def _reduction_keepdim(node: ast.Call, positional_index: int) -> bool:
    """Extract the ``keepdim`` flag from a reduction call (keyword or positional bool)."""
    for kw in node.keywords:
        if kw.arg == "keepdim":
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                return kw.value.value
    if len(node.args) > positional_index:
        arg = node.args[positional_index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, bool):
            return arg.value
    return False


def _shapes_definitely_mismatch(declared: TensorShape, actual: TensorShape) -> bool:
    """Return True only when the shapes are provably incompatible."""
    if declared.rank != actual.rank:
        return True
    for d, a in zip(declared.dims, actual.dims, strict=True):
        if isinstance(d, ConstantDim) and isinstance(a, ConstantDim) and d.value != a.value:
            return True
        if isinstance(d, SymbolicDim) and isinstance(a, SymbolicDim) and d.name != a.name:
            return True
        if isinstance(d, ConstantDim) and isinstance(a, SymbolicDim):
            return True
        if isinstance(d, SymbolicDim) and isinstance(a, ConstantDim):
            return True
    return False


def _size_to_dim(
    node: ast.expr,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> Dim:
    """Convert a size expression to a Dim; never returns None."""
    val = int_from_ast(node)
    if val is not None:
        return ConstantDim(val)
    if isinstance(node, ast.Name):
        env_val = env.get(node.id)
        if isinstance(env_val, IntegerValue) and env_val.value is not None:
            return ConstantDim(env_val.value)
        return UnknownDim(node.id)
    result = _eval_expr(node, env, context, module_specs)
    if isinstance(result, IntegerValue) and result.value is not None:
        return ConstantDim(result.value)
    if isinstance(result, (ConstantDim, SymbolicDim, ExpressionDim, UnknownDim)):
        return result
    return UnknownDim("?")


def _constructor_size(
    node: ast.Call,
    constructor: str,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> list[Dim] | None:
    """Extract size dimensions from a tensor constructor call."""
    if constructor == "full":
        if not node.args:
            return None
        size_arg = node.args[0]
        if isinstance(size_arg, (ast.Tuple, ast.List)):
            return [_size_to_dim(e, env, context, module_specs) for e in size_arg.elts]
        return [_size_to_dim(size_arg, env, context, module_specs)]
    if not node.args:
        for kw in node.keywords:
            if kw.arg == "size" and isinstance(kw.value, (ast.Tuple, ast.List)):
                return [_size_to_dim(e, env, context, module_specs) for e in kw.value.elts]
        return None
    if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)):
        return [_size_to_dim(e, env, context, module_specs) for e in node.args[0].elts]
    return [_size_to_dim(a, env, context, module_specs) for a in node.args]


def _arange_length(node: ast.Call) -> int | None:
    """Return the number of elements in a ``torch.arange`` call if constant."""
    if len(node.args) == 1:
        return int_from_ast(node.args[0])
    if len(node.args) == 2:
        start = int_from_ast(node.args[0])
        end = int_from_ast(node.args[1])
        if start is not None and end is not None and end >= start:
            return end - start
    if len(node.args) == 3:
        start = int_from_ast(node.args[0])
        end = int_from_ast(node.args[1])
        step = int_from_ast(node.args[2])
        if start is not None and end is not None and step is not None and step > 0:
            return (end - start + step - 1) // step
    return None


def _infer_expand(
    tensor: TensorValue,
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> TensorValue:
    """Infer output shape of ``x.expand(*sizes)``."""
    size_nodes: list[ast.expr] = list(node.args)
    if len(size_nodes) == 1 and isinstance(size_nodes[0], (ast.Tuple, ast.List)):
        size_nodes = list(size_nodes[0].elts)
    n = len(size_nodes)
    rank = tensor.rank
    result_dims: list[Dim] = []
    for i, size_node in enumerate(size_nodes):
        orig_idx = i - (n - rank)
        val = int_from_ast(size_node)
        if val == -1 and 0 <= orig_idx < rank:
            result_dims.append(tensor.shape.dims[orig_idx])
        else:
            result_dims.append(_size_to_dim(size_node, env, context, module_specs))
    return TensorValue(TensorShape(tuple(result_dims)))


def _infer_repeat(
    tensor: TensorValue,
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> TensorValue:
    """Infer output shape of ``x.repeat(*repeats)``."""
    size_nodes: list[ast.expr] = list(node.args)
    if len(size_nodes) == 1 and isinstance(size_nodes[0], (ast.Tuple, ast.List)):
        size_nodes = list(size_nodes[0].elts)
    n = len(size_nodes)
    rank = tensor.rank
    if n < rank:
        return tensor
    padded: tuple[Dim, ...] = (ConstantDim(1),) * (n - rank) + tensor.shape.dims
    result_dims: list[Dim] = []
    for d, size_node in zip(padded, size_nodes, strict=True):
        repeat_val = int_from_ast(size_node)
        if repeat_val is not None:
            if isinstance(d, ConstantDim):
                result_dims.append(ConstantDim(d.value * repeat_val))
            elif repeat_val == 1:
                result_dims.append(d)
            else:
                result_dims.append(ExpressionDim(f"{d}*{repeat_val}"))
        else:
            result_dims.append(ExpressionDim(f"{d}*?"))
    return TensorValue(TensorShape(tuple(result_dims)))


def _split_from_call(tensor: TensorValue, node: ast.Call) -> TensorTupleValue | None:
    """Parse split arguments from a ``.split(size_or_sections, dim)`` call."""
    if not node.args:
        return None
    size_node = node.args[0]
    dim = _positional_int(node.args, 1, None)
    if dim is None:
        dim = _keyword_int(node, "dim", 0)
    if dim is None:
        dim = 0
    if isinstance(size_node, (ast.List, ast.Tuple)):
        sizes = [int_from_ast(e) for e in size_node.elts]
        if any(s is None for s in sizes):
            return None
        return infer_split(tensor, [s for s in sizes if s is not None], dim)
    split_size = int_from_ast(size_node)
    if split_size is not None:
        return infer_split(tensor, split_size, dim)
    return None


def _interpolate_size_arg(
    node: ast.Call,
    n_spatial: int,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> tuple[Dim, ...] | None:
    """Extract the ``size`` argument from an ``F.interpolate`` call."""
    size_node: ast.AST | None = None
    if len(node.args) >= 2:
        size_node = node.args[1]
    else:
        size_node = _keyword_or_default(node, "size")
    if size_node is None:
        return None
    # Variable reference (e.g. label.shape[-2:]) → evaluate as ShapeTupleValue.
    val = _eval_expr(size_node, env, context, module_specs)
    if isinstance(val, ShapeTupleValue):
        dims = val.dims
        return tuple(dims[-n_spatial:]) if len(dims) >= n_spatial else None
    # Single int constant → replicate for all spatial dims.
    single = int_from_ast(size_node)
    if single is not None:
        return tuple(ConstantDim(single) for _ in range(n_spatial))
    # Tuple/list of int constants.
    if isinstance(size_node, (ast.Tuple, ast.List)):
        result_dims: list[Dim] = []
        for elt in size_node.elts:
            v = int_from_ast(elt)
            result_dims.append(ConstantDim(v) if v is not None else UnknownDim("?"))
        return tuple(result_dims)
    # Unknown.
    return tuple(UnknownDim("?") for _ in range(n_spatial))


def _interpolate_scale_arg(node: ast.Call, n_spatial: int) -> tuple[float, ...] | None:
    """Extract the ``scale_factor`` argument from an ``F.interpolate`` call."""
    scale_node = _keyword_or_default(node, "scale_factor")
    if scale_node is None:
        return None
    if isinstance(scale_node, ast.Constant) and isinstance(scale_node.value, (int, float)):
        f = float(scale_node.value)
        return tuple(f for _ in range(n_spatial))
    if isinstance(scale_node, (ast.Tuple, ast.List)):
        factors: list[float] = []
        for elt in scale_node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                factors.append(float(elt.value))
            else:
                return None
        return tuple(factors)
    return None

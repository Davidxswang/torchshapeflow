from __future__ import annotations

import ast
import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from torchshapeflow.diagnostics import Diagnostic, Severity, render_message
from torchshapeflow.index import (
    CustomModuleTemplate,
    FuncSig,
    ProjectIndex,
    apply_substitution,
    build_file_data,
    extract_alias_binding,
    extract_func_sig,
    unify_dims,
)
from torchshapeflow.model import (
    ConstantDim,
    Conv2dSpec,
    CustomModuleSpec,
    Dim,
    EmbeddingSpec,
    ExpressionDim,
    IntegerValue,
    LinearSpec,
    LSTMSpec,
    ModuleSpec,
    MultiheadAttentionSpec,
    PassthroughSpec,
    Pool2dSpec,
    RepeatSpec,
    SequentialSpec,
    ShapeTupleValue,
    SymbolicDim,
    TensorShape,
    TensorTupleValue,
    TensorValue,
    TupleValue,
    UnknownDim,
    Value,
    broadcast_has_uncertain_dims,
    make_dim,
    normalize_index,
    product_dim,
    quotient_dim,
    render_dim,
    sum_dim,
)
from torchshapeflow.parser import AnnotationParseError, parse_source, parse_tensor_annotation
from torchshapeflow.report import FileReport, HoverFact, Suggestion
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
    infer_lstm,
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

_MODULE_SPEC_TYPES = (
    LinearSpec,
    Conv2dSpec,
    PassthroughSpec,
    EmbeddingSpec,
    Pool2dSpec,
    CustomModuleSpec,
    RepeatSpec,
    SequentialSpec,
    MultiheadAttentionSpec,
    LSTMSpec,
)


@dataclass
class ModuleContext:
    path: Path
    diagnostics: list[Diagnostic] = field(default_factory=list)
    hovers: list[HoverFact] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)
    aliases: dict[str, TensorValue] = field(default_factory=dict)
    func_sigs: dict[str, FuncSig] = field(default_factory=dict)
    return_shape: TensorValue | None = None
    collected_returns: list[TensorValue | None] = field(default_factory=list)
    in_annotated_function: bool = False
    # Scalar self.attr values from __init__ (e.g. self.hidden = hidden_dim → SymbolicDim).
    self_scalars: dict[str, Dim] = field(default_factory=dict)
    # Tensor self.attr values captured from __init__ (e.g. register_buffer or direct assignment).
    self_tensors: dict[str, TensorValue] = field(default_factory=dict)
    # Method signatures for the current class being analyzed.
    method_sigs: dict[str, FuncSig] = field(default_factory=dict)

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

    def shape_error(
        self,
        node: ast.AST,
        code: str,
        summary: str,
        *,
        expected: str | None = None,
        actual: str | None = None,
        hint: str | None = None,
        severity: Severity = "error",
    ) -> None:
        """Append a shape-mismatch diagnostic with structured fields.

        Structured fields are the source of truth; the human-readable message
        is rendered from them via ``render_message`` to keep prose and data in
        sync. Agents and editors can consume ``expected`` / ``actual`` /
        ``hint`` directly from JSON output.
        """
        self.diagnostics.append(
            Diagnostic(
                code=code,
                message=render_message(summary, expected=expected, actual=actual, hint=hint),
                path=self.path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                severity=severity,
                expected=expected,
                actual=actual,
                hint=hint,
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

    def hover_alias(self, name: str, node: ast.AST, tensor: TensorValue) -> None:
        self.hovers.append(
            HoverFact(
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                end_column=getattr(node, "end_col_offset", getattr(node, "col_offset", 0)) + 1,
                name=name,
                shape=str(tensor.shape),
                kind="alias",
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
    file_data = build_file_data(module, path, project_index)
    context = ModuleContext(path=path, aliases=file_data.aliases, func_sigs=file_data.func_sigs)
    _emit_module_alias_hovers(module, context)
    class_specs, class_scalars, class_tensors = _collect_class_specs(
        module, context, file_data.custom_module_templates
    )
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            _analyze_function(node, context, {})
        elif isinstance(node, ast.ClassDef):
            specs = class_specs.get(node.name, {})
            context.self_scalars = class_scalars.get(node.name, {})
            context.self_tensors = class_tensors.get(node.name, {})
            # Pass 1: Collect method signatures for self.method() lookups.
            context.method_sigs = {}
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    sig = extract_func_sig(child, context.aliases)
                    if sig is not None:
                        context.method_sigs[child.name] = sig
            # Pass 2: Analyze all methods (except __init__ which is handled in specs).
            for child in node.body:
                if not isinstance(child, ast.FunctionDef):
                    continue
                if child.name == "__init__":
                    _emit_function_annotation_hovers(child, context)
                    continue
                _analyze_function(child, context, specs)
            context.method_sigs = {}
            context.self_scalars = {}
            context.self_tensors = {}
    return FileReport(
        path=str(path),
        diagnostics=context.diagnostics,
        hovers=context.hovers,
        suggestions=context.suggestions,
    )


def _collect_class_specs(
    module: ast.Module,
    context: ModuleContext,
    custom_module_templates: dict[str, CustomModuleTemplate],
) -> tuple[
    dict[str, dict[str, ModuleSpec]],
    dict[str, dict[str, Dim]],
    dict[str, dict[str, TensorValue]],
]:
    specs: dict[str, dict[str, ModuleSpec]] = {}
    scalars: dict[str, dict[str, Dim]] = {}
    tensors: dict[str, dict[str, TensorValue]] = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_specs: dict[str, ModuleSpec] = {}
        class_scalars: dict[str, Dim] = {}
        class_tensors: dict[str, TensorValue] = {}
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                init_env, _, _, _ = _collect_function_annotations(child, context, emit_hovers=False)
                positive_params = _collect_positive_scalar_params(child)
                sequence_specs: dict[str, SequentialSpec] = {}
                for statement in child.body:
                    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                        target = statement.targets[0]
                        if (
                            isinstance(target, ast.Name)
                            and isinstance(statement.value, ast.List)
                            and len(statement.value.elts) == 0
                        ):
                            sequence_specs[target.id] = SequentialSpec(specs=())
                            continue
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            spec = _parse_module_spec(
                                statement.value, sequence_specs, custom_module_templates
                            )
                            if spec is not None:
                                class_specs[target.attr] = spec
                            else:
                                tensor_value = _init_tensor_from_expr(statement.value, init_env)
                                if tensor_value is not None:
                                    class_tensors[target.attr] = tensor_value
                            if isinstance(statement.value, ast.Name):
                                # self.attr = variable_name → track as SymbolicDim
                                class_scalars[target.attr] = SymbolicDim(statement.value.id)
                        elif isinstance(target, ast.Name):
                            value = _init_tensor_from_expr(statement.value, init_env)
                            if value is not None:
                                init_env[target.id] = value
                    elif (
                        isinstance(statement, ast.AnnAssign)
                        and isinstance(statement.target, ast.Name)
                        and isinstance(statement.value, ast.List)
                        and len(statement.value.elts) == 0
                    ):
                        sequence_specs[statement.target.id] = SequentialSpec(specs=())
                    elif isinstance(statement, ast.For):
                        sequence_binding = _loop_sequence_binding(
                            statement, sequence_specs, positive_params, custom_module_templates
                        )
                        if sequence_binding is not None:
                            name, spec = sequence_binding
                            sequence_specs[name] = spec
                    elif _is_register_buffer_call(statement):
                        buffer = _register_buffer_binding(statement, init_env)
                        if buffer is not None:
                            name, tensor_value = buffer
                            class_tensors[name] = tensor_value
        if class_specs:
            specs[node.name] = class_specs
        if class_scalars:
            scalars[node.name] = class_scalars
        if class_tensors:
            tensors[node.name] = class_tensors
    return specs, scalars, tensors


def _emit_module_alias_hovers(module: ast.Module, context: ModuleContext) -> None:
    for statement in module.body:
        alias = extract_alias_binding(statement)
        if alias is None:
            continue
        alias_name, alias_node = alias
        tensor = context.aliases.get(alias_name)
        target = _alias_target_node(statement)
        if tensor is not None and target is not None:
            context.hover_alias(alias_name, target, tensor)


def _parse_module_spec(
    node: ast.AST,
    sequence_specs: dict[str, SequentialSpec] | None = None,
    custom_module_templates: dict[str, CustomModuleTemplate] | None = None,
) -> ModuleSpec | None:
    if isinstance(node, ast.Name) and sequence_specs is not None:
        return sequence_specs.get(node.id)
    if not isinstance(node, ast.Call):
        return None
    name = qualified_name(node.func)
    short_name = name.split(".")[-1]

    def _int(n: ast.AST) -> int | None:
        """Resolve a literal integer from an AST node (input/validation dims only)."""
        return int_from_ast(n)

    def _sym(n: ast.AST) -> int | str | None:
        """Resolve an output dim: literal int, variable name, or simple Name op Name expr."""
        val = int_from_ast(n)
        if val is not None:
            return val
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Add, ast.Sub)):
            left = _sym(n.left)
            right = _sym(n.right)
            if left is not None and right is not None:
                if isinstance(left, int) and isinstance(right, int):
                    return left  # already handled by int_from_ast; unreachable in practice
                if isinstance(n.op, ast.Mult):
                    op_str = "*"
                elif isinstance(n.op, ast.Add):
                    op_str = "+"
                else:
                    op_str = "-"
                return f"{left}{op_str}{right}"
        return None

    if name.endswith("Linear") and len(node.args) >= 2:
        in_features = _int(node.args[0])  # may be None (non-literal)
        out_features = _sym(node.args[1])
        if out_features is not None:
            return LinearSpec(in_features=in_features, out_features=out_features)
    if name.endswith("Conv2d") and len(node.args) >= 3:
        in_channels = _int(node.args[0])  # may be None (non-literal)
        out_channels = _sym(node.args[1])
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
        embedding_dim = _sym(node.args[1])
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
            sub_node = arg.value if isinstance(arg, ast.Starred) else arg
            sub = _parse_module_spec(sub_node, sequence_specs, custom_module_templates)
            if sub is None:
                return None
            sub_specs.append(sub)
        return SequentialSpec(specs=tuple(sub_specs))
    if short_name == "ModuleList" and len(node.args) == 1:
        return _parse_module_spec(node.args[0], sequence_specs, custom_module_templates)
    if custom_module_templates is not None:
        template = custom_module_templates.get(short_name)
        if template is not None:
            return _bind_custom_module_template(node, template)
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
    if short_name == "LSTM" and (len(node.args) >= 2 or _keyword_or_default(node, "hidden_size")):
        input_size_node = (
            node.args[0] if len(node.args) >= 1 else _keyword_or_default(node, "input_size")
        )
        hidden_size_node = (
            node.args[1] if len(node.args) >= 2 else _keyword_or_default(node, "hidden_size")
        )
        if hidden_size_node is None:
            return None
        input_size = _int(input_size_node) if input_size_node is not None else None
        hidden_size = _sym(hidden_size_node)
        proj_size: int | str | None = None
        proj_size_arg = _keyword_or_default(node, "proj_size")
        if proj_size_arg is not None:
            proj_size = _sym(proj_size_arg)
        num_layers: int | str = 1
        num_layers_arg = (
            node.args[2] if len(node.args) >= 3 else _keyword_or_default(node, "num_layers")
        )
        if num_layers_arg is not None:
            parsed = _sym(num_layers_arg)
            if parsed is not None:
                num_layers = parsed
        batch_first_node = _keyword_or_default(node, "batch_first")
        batch_first = False
        if isinstance(batch_first_node, ast.Constant) and isinstance(batch_first_node.value, bool):
            batch_first = batch_first_node.value
        bidirectional_node = _keyword_or_default(node, "bidirectional")
        bidirectional = False
        if isinstance(bidirectional_node, ast.Constant) and isinstance(
            bidirectional_node.value, bool
        ):
            bidirectional = bidirectional_node.value
        if hidden_size is not None:
            return LSTMSpec(
                input_size=input_size,
                hidden_size=hidden_size,
                proj_size=proj_size,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
            )
    return None


def _bind_custom_module_template(
    call: ast.Call,
    template: CustomModuleTemplate,
) -> CustomModuleSpec:
    mapping: dict[str, Dim] = {}
    for index, param_name in enumerate(template.init_param_names):
        arg_node: ast.AST | None
        if index < len(call.args):
            arg_node = call.args[index]
        else:
            arg_node = _keyword_or_default(call, param_name)
        dim_value = _dim_symbol_from_ast(arg_node) if arg_node is not None else None
        if dim_value is not None:
            mapping[param_name] = make_dim(dim_value)

    input_shape = _substitute_tensor_value(template.input_shape, mapping)
    return_shape = _substitute_tensor_value(template.return_shape, mapping)
    return CustomModuleSpec(input_shape=input_shape, return_shape=return_shape)


def _substitute_tensor_value(
    tensor: TensorValue | None, mapping: dict[str, Dim]
) -> TensorValue | None:
    if tensor is None:
        return None
    return TensorValue(apply_substitution(tensor.shape, mapping), origin=tensor.origin)


def _dim_symbol_from_ast(node: ast.AST | None) -> int | str | None:
    if node is None:
        return None
    val = int_from_ast(node)
    if val is not None:
        return val
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        left = _dim_symbol_from_ast(node.left)
        right = _dim_symbol_from_ast(node.right)
        if left is None or right is None:
            return None
        if isinstance(left, int) and isinstance(right, int):
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Add):
                return left + right
            return left - right
        if isinstance(node.op, ast.Mult):
            op_str = "*"
        elif isinstance(node.op, ast.Add):
            op_str = "+"
        else:
            op_str = "-"
        return f"{left}{op_str}{right}"
    return None


def _collect_positive_scalar_params(init_func: ast.FunctionDef) -> set[str]:
    positive: set[str] = set()
    for statement in init_func.body:
        if not isinstance(statement, ast.If):
            continue
        test = statement.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and len(test.comparators) == 1
            and isinstance(test.left, ast.Name)
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, int)
            and any(isinstance(item, ast.Raise) for item in statement.body)
        ):
            op = test.ops[0]
            bound = test.comparators[0].value
            if (isinstance(op, ast.LtE) and bound == 0) or (isinstance(op, ast.Lt) and bound == 1):
                positive.add(test.left.id)
    return positive


def _loop_sequence_binding(
    statement: ast.For,
    sequence_specs: dict[str, SequentialSpec],
    positive_params: set[str],
    custom_module_templates: dict[str, CustomModuleTemplate],
) -> tuple[str, SequentialSpec] | None:
    if not isinstance(statement.target, ast.Name):
        return None
    if not (
        isinstance(statement.iter, ast.Call)
        and isinstance(statement.iter.func, ast.Name)
        and statement.iter.func.id == "range"
        and len(statement.iter.args) == 1
    ):
        return None
    count_node = statement.iter.args[0]
    count_literal = int_from_ast(count_node)
    count_name = count_node.id if isinstance(count_node, ast.Name) else None
    min_count = (
        1
        if (count_literal is not None and count_literal > 0) or count_name in positive_params
        else 0
    )

    loop_var = statement.target.id
    append_target: str | None = None
    append_arg: ast.expr | None = None
    local_bindings: dict[str, ast.expr] = {}
    for body_stmt in statement.body:
        if isinstance(body_stmt, ast.Assign) and len(body_stmt.targets) == 1:
            target = body_stmt.targets[0]
            if isinstance(target, ast.Name):
                local_bindings[target.id] = body_stmt.value
            continue
        if (
            isinstance(body_stmt, ast.Expr)
            and isinstance(body_stmt.value, ast.Call)
            and isinstance(body_stmt.value.func, ast.Attribute)
            and isinstance(body_stmt.value.func.value, ast.Name)
            and body_stmt.value.func.attr == "append"
            and len(body_stmt.value.args) == 1
        ):
            append_target = body_stmt.value.func.value.id
            append_arg = body_stmt.value.args[0]

    if append_target is None or append_arg is None or append_target not in sequence_specs:
        return None

    first_node = _resolve_loop_expr(append_arg, local_bindings, loop_var, is_first=True)
    rest_node = _resolve_loop_expr(append_arg, local_bindings, loop_var, is_first=False)
    first_spec = _parse_module_spec(first_node, sequence_specs, custom_module_templates)
    rest_spec = _parse_module_spec(rest_node, sequence_specs, custom_module_templates)
    if first_spec is None and rest_spec is None:
        return None

    # Exact literal trip count: emit a finite summary.
    if count_literal is not None:
        if count_literal <= 0:
            return append_target, SequentialSpec(specs=())
        if first_spec is None:
            return None
        if count_literal == 1:
            return append_target, SequentialSpec(specs=(first_spec,))
        if rest_spec is None:
            return None
        if first_spec == rest_spec:
            return append_target, SequentialSpec(
                specs=(RepeatSpec(first_spec, count_literal, min_count=count_literal),)
            )
        return append_target, SequentialSpec(
            specs=(
                first_spec,
                RepeatSpec(rest_spec, count_literal - 1, min_count=count_literal - 1),
            )
        )

    # Unknown count with no positivity guarantee: only safe when every iteration has the same spec.
    if first_spec is not None and rest_spec is not None and first_spec == rest_spec:
        return append_target, SequentialSpec(
            specs=(RepeatSpec(first_spec, count_name, min_count=min_count),)
        )

    # Unknown positive count: emit one required first stage plus an unknown optional stable suffix.
    if min_count >= 1 and first_spec is not None and rest_spec is not None:
        return append_target, SequentialSpec(
            specs=(first_spec, RepeatSpec(rest_spec, None, min_count=0))
        )

    return None


def _resolve_loop_expr(
    node: ast.expr,
    bindings: dict[str, ast.expr],
    loop_var: str,
    *,
    is_first: bool,
) -> ast.expr:
    if isinstance(node, ast.Name) and node.id in bindings:
        return _resolve_loop_expr(bindings[node.id], bindings, loop_var, is_first=is_first)
    if isinstance(node, ast.IfExp) and _is_first_iteration_test(node.test, loop_var):
        branch = node.body if is_first else node.orelse
        return _resolve_loop_expr(branch, bindings, loop_var, is_first=is_first)
    if isinstance(node, ast.Call):
        return ast.Call(
            func=_resolve_loop_expr(node.func, bindings, loop_var, is_first=is_first),
            args=[
                _resolve_loop_expr(arg, bindings, loop_var, is_first=is_first) for arg in node.args
            ],
            keywords=[
                ast.keyword(
                    arg=kw.arg,
                    value=_resolve_loop_expr(kw.value, bindings, loop_var, is_first=is_first),
                )
                for kw in node.keywords
            ],
        )
    if isinstance(node, ast.BinOp):
        return ast.BinOp(
            left=_resolve_loop_expr(node.left, bindings, loop_var, is_first=is_first),
            op=node.op,
            right=_resolve_loop_expr(node.right, bindings, loop_var, is_first=is_first),
        )
    if isinstance(node, ast.Attribute):
        return ast.Attribute(
            value=_resolve_loop_expr(node.value, bindings, loop_var, is_first=is_first),
            attr=node.attr,
            ctx=node.ctx,
        )
    return node


def _is_first_iteration_test(test: ast.AST, loop_var: str) -> bool:
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = test.left
    right = test.comparators[0]
    return isinstance(test.ops[0], ast.Eq) and (
        (
            isinstance(left, ast.Name)
            and left.id == loop_var
            and isinstance(right, ast.Constant)
            and right.value == 0
        )
        or (
            isinstance(right, ast.Name)
            and right.id == loop_var
            and isinstance(left, ast.Constant)
            and left.value == 0
        )
    )


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
    env, local_aliases, tensor_params, return_shape = _collect_function_annotations(
        function, context
    )
    # Track whether this function has tensor-annotated parameters (for TSF2xxx warnings).
    old_in_annotated = context.in_annotated_function
    context.in_annotated_function = len(tensor_params) > 0
    # Parse return annotation; set on context so _analyze_statement can validate.
    old_return_shape = context.return_shape
    old_collected_returns = context.collected_returns
    context.collected_returns = []
    context.return_shape = return_shape
    # Snapshot error-severity diagnostic count so the suggest helper can tell
    # whether analyzing this function surfaced any errors.
    errors_before = sum(1 for d in context.diagnostics if d.severity == "error")
    for statement in function.body:
        _analyze_statement(statement, env, context, module_specs, local_aliases)
    # Emit a signature hover on the function name if any tensor params are present.
    if tensor_params:
        _emit_signature_hover(
            function, tensor_params, context.return_shape, context.collected_returns, context
        )
    _maybe_suggest_return_annotation(
        function,
        tensor_params,
        context.return_shape,
        context.collected_returns,
        context,
        errors_before=errors_before,
    )
    context.collected_returns = old_collected_returns
    context.return_shape = old_return_shape
    context.in_annotated_function = old_in_annotated


def _emit_function_annotation_hovers(function: ast.FunctionDef, context: ModuleContext) -> None:
    _, _, tensor_params, return_shape = _collect_function_annotations(function, context)
    if tensor_params:
        _emit_signature_hover(function, tensor_params, return_shape, [], context)


def _contains_top_level_yield(body: list[ast.stmt]) -> bool:
    """True iff *body* contains a ``yield`` or ``yield from`` at its own scope.

    A ``yield`` in the outer function makes that function a generator — calling
    it returns a ``Generator[...]`` object, never the tensor the ``return``
    statement names (which becomes the ``StopIteration`` value). We must not
    propose a plain-tensor return annotation for generators.

    Walks the statement tree but does not descend into nested ``def``,
    ``async def``, or ``lambda`` bodies: a yield inside one of those makes the
    inner callable a generator, not the outer one.
    """
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _body_terminates_with_return(body: list[ast.stmt]) -> bool:
    """True iff *body* provably ends by returning a value.

    Recognizes:
    - A trailing ``return X`` (with value) at the end of the body.
    - A trailing ``raise`` (function exits without falling through to None).
    - A trailing ``if`` with ``else`` where every branch terminates.

    Loops, ``try``/``except``, ``match``, and bare ``return`` (no value) yield
    False — an honest "don't know" that keeps ``tsf suggest`` silent rather
    than asserting a shape contract the analyzer cannot prove.
    """
    if not body:
        return False
    last = body[-1]
    if isinstance(last, ast.Return) and last.value is not None:
        return True
    if isinstance(last, ast.Raise):
        return True
    if isinstance(last, ast.If) and last.orelse:
        return _body_terminates_with_return(last.body) and _body_terminates_with_return(last.orelse)
    return False


def _first_annotated_param_template(
    function: ast.FunctionDef,
    tensor_param_names: set[str],
) -> ast.expr | None:
    """Return the annotation AST of the first param recognized as a Shape contract.

    That annotation is already known to parse (the caller filtered to params
    that produced a TensorValue), so its constituent names — ``Annotated``,
    the tensor type, ``Shape`` — all resolve in the target file. The
    suggestion renderer reuses this AST as a style template, guaranteeing
    that the proposed annotation uses only in-scope names.
    """
    for arg in function.args.args:
        if arg.arg in tensor_param_names and arg.annotation is not None:
            return arg.annotation
    return None


def _rebuild_shape_metadata(
    original: ast.expr,
    shape: TensorShape,
) -> ast.expr | None:
    """Rebuild a ``Shape(...)`` call or string-shorthand with *shape*'s dims.

    Returns None when *original* is neither form, or when any dim cannot be
    expressed as a parser-accepted arg (``ExpressionDim`` / ``UnknownDim``).
    """
    if isinstance(original, ast.Call):
        new_args: list[ast.expr] = []
        for dim in shape.dims:
            if isinstance(dim, ConstantDim):
                new_args.append(ast.Constant(value=dim.value))
            elif isinstance(dim, SymbolicDim):
                new_args.append(ast.Constant(value=dim.name))
            else:
                return None
        # Preserve original.func so spellings like ``torchshapeflow.Shape(...)``
        # survive — we only swap the args.
        return ast.Call(
            func=copy.deepcopy(original.func),
            args=new_args,
            keywords=[],
        )
    if isinstance(original, ast.Constant) and isinstance(original.value, str):
        parts: list[str] = []
        for dim in shape.dims:
            if isinstance(dim, ConstantDim):
                parts.append(str(dim.value))
            elif isinstance(dim, SymbolicDim):
                parts.append(dim.name)
            else:
                return None
        return ast.Constant(value=" ".join(parts))
    return None


def _render_return_annotation_from_template(
    template: ast.expr,
    shape: TensorShape,
) -> str | None:
    """Render an ``Annotated[..., Shape(...)]`` source string for *shape*,
    reusing *template*'s spelling so the suggestion refers only to names the
    target file has already imported.

    Returns None when the template isn't an ``Annotated[...]`` subscript with
    a rewritable metadata slot (``Shape(...)`` call or string shorthand). This
    includes TypeAlias-annotated params (``x: Batch``) — under the
    propose-don't-decide principle, skipping is strictly better than emitting
    source that may not parse in the caller's file.
    """
    if not isinstance(template, ast.Subscript):
        return None
    slice_node = template.slice
    if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) < 2:
        return None
    new_metadata = _rebuild_shape_metadata(slice_node.elts[1], shape)
    if new_metadata is None:
        return None
    rebuilt = copy.deepcopy(template)
    assert isinstance(rebuilt, ast.Subscript)
    assert isinstance(rebuilt.slice, ast.Tuple)
    rebuilt.slice.elts[1] = new_metadata
    return ast.unparse(rebuilt)


def _maybe_suggest_return_annotation(
    function: ast.FunctionDef,
    tensor_params: list[tuple[str, TensorValue]],
    declared_return: TensorValue | None,
    collected_returns: list[TensorValue | None],
    context: ModuleContext,
    *,
    errors_before: int,
) -> None:
    """Propose a return annotation when the analyzer can verify the shape.

    Emits a suggestion only when every precondition holds:

    - At least one parameter has a ``Shape`` annotation (user opted in).
    - The function has no return annotation at all (``function.returns`` is None).
    - Analyzing the function body added no new error-severity diagnostics —
      TSF must not propose a contract on code it has also flagged as broken.
    - ``_body_terminates_with_return`` proves every exit path returns a value
      (guards against implicit fallthrough → None and bare ``return``).
    - Every collected return expression produced a ``TensorValue`` with the
      same shape.
    - The shape is expressible in ``Shape(...)`` syntax (no ExpressionDim /
      UnknownDim).
    - ``_render_return_annotation_from_template`` can reuse the first
      annotated param's spelling (guards against names not in scope).

    Skipping when any precondition fails is intentional: under the
    propose-don't-decide principle, missing a legitimate suggestion is
    strictly better than emitting one that is false or won't parse.
    """
    if not tensor_params:
        return
    if function.returns is not None:
        return
    if declared_return is not None:
        return
    if not collected_returns:
        return
    if any(r is None for r in collected_returns):
        return
    unique_shapes = {str(r.shape) for r in collected_returns if r is not None}
    if len(unique_shapes) != 1:
        return
    errors_after = sum(1 for d in context.diagnostics if d.severity == "error")
    if errors_after > errors_before:
        return
    if _contains_top_level_yield(function.body):
        return
    if not _body_terminates_with_return(function.body):
        return
    inferred = next(r for r in collected_returns if r is not None)
    template = _first_annotated_param_template(function, {name for name, _ in tensor_params})
    if template is None:
        return
    annotation = _render_return_annotation_from_template(template, inferred.shape)
    if annotation is None:
        return
    name_col = function.col_offset + 4
    name_end_col = name_col + len(function.name)
    context.suggestions.append(
        Suggestion(
            line=function.lineno,
            column=name_col + 1,
            end_line=function.lineno,
            end_column=name_end_col + 1,
            function=function.name,
            shape=str(inferred.shape),
            annotation=annotation,
        )
    )


def _collect_function_annotations(
    function: ast.FunctionDef,
    context: ModuleContext,
    *,
    emit_hovers: bool = True,
) -> tuple[
    dict[str, Value], dict[str, TensorValue], list[tuple[str, TensorValue]], TensorValue | None
]:
    env: dict[str, Value] = {}
    local_aliases: dict[str, TensorValue] = dict(context.aliases)
    tensor_params: list[tuple[str, TensorValue]] = []
    for argument in function.args.args:
        if argument.arg == "self":
            continue
        if argument.annotation is None:
            continue
        try:
            tensor = parse_tensor_annotation(argument.annotation, local_aliases)
        except AnnotationParseError as error:
            context.error(argument, "TSF1001", error.message)
            continue
        if emit_hovers:
            _maybe_hover_alias_reference(argument.annotation, local_aliases, context)
        if tensor is not None:
            env[argument.arg] = tensor
            if emit_hovers:
                context.hover(argument.arg, argument, tensor)
            tensor_params.append((argument.arg, tensor))

    return_shape: TensorValue | None = None
    if function.returns is not None:
        try:
            return_shape = parse_tensor_annotation(function.returns, local_aliases)
        except AnnotationParseError:
            return_shape = None
        if emit_hovers:
            _maybe_hover_alias_reference(function.returns, local_aliases, context)

    return env, local_aliases, tensor_params, return_shape


def _init_tensor_from_expr(node: ast.AST, env: dict[str, Value]) -> TensorValue | None:
    value = env.get(node.id) if isinstance(node, ast.Name) else None
    if isinstance(value, TensorValue):
        return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = _init_tensor_from_expr(node.func.value, env)
        if base is not None and node.func.attr in _PASSTHROUGH_METHODS:
            return base
    return None


def _is_register_buffer_call(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == "register_buffer"
    )


def _register_buffer_binding(
    statement: ast.stmt,
    env: dict[str, Value],
) -> tuple[str, TensorValue] | None:
    if not _is_register_buffer_call(statement):
        return None
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    if len(call.args) < 2:
        return None
    name_node = call.args[0]
    if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
        return None
    tensor_value = _init_tensor_from_expr(call.args[1], env)
    if tensor_value is None:
        return None
    return name_node.value, tensor_value


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
            kind="signature",
        )
    )


def _unpack_tensor_tuple(
    elts: list[ast.expr],
    values: tuple[Value, ...],
    env: dict[str, Value],
    context: ModuleContext,
) -> None:
    """Bind names from a tuple unpack against a statically-known tuple value.

    Handles both flat and nested patterns, e.g.:
      ``a, b, c = chunk_result``          — flat
      ``out, (h, c) = lstm(x)``           — nested tuple structure
      ``_, (hidden, _) = lstm(x)``        — nested with wildcards
    """
    value_idx = 0
    for elt in elts:
        if value_idx >= len(values):
            break
        current = values[value_idx]
        if isinstance(elt, ast.Tuple):
            if isinstance(current, TensorTupleValue):
                _unpack_tensor_tuple(list(elt.elts), current.tensors, env, context)
            elif isinstance(current, TupleValue):
                _unpack_tensor_tuple(list(elt.elts), current.items, env, context)
            else:
                _bind_target(elt, current, env, context)
        else:
            _bind_target(elt, current, env, context)
        value_idx += 1


def _analyze_statement(
    statement: ast.stmt,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
    aliases: dict[str, TensorValue],
) -> None:
    alias = extract_alias_binding(statement)
    if alias is not None:
        alias_name, alias_node = alias
        try:
            alias_value = parse_tensor_annotation(alias_node, aliases)
        except AnnotationParseError as error:
            context.error(statement, "TSF1001", error.message)
            return
        if alias_value is not None:
            aliases[alias_name] = alias_value
            target = _alias_target_node(statement)
            if target is not None:
                context.hover_alias(alias_name, target, alias_value)
            return
        if isinstance(statement, ast.AnnAssign) or (
            hasattr(ast, "TypeAlias") and isinstance(statement, ast.TypeAlias)
        ):
            context.error(
                statement,
                "TSF1001",
                "TypeAlias must resolve to an Annotated tensor annotation.",
            )
            return
    if isinstance(statement, ast.Assign):
        value = _eval_expr(statement.value, env, context, module_specs)
        for assign_target in statement.targets:
            if isinstance(assign_target, ast.Tuple) and isinstance(value, TensorTupleValue):
                _unpack_tensor_tuple(list(assign_target.elts), value.tensors, env, context)
            elif isinstance(assign_target, ast.Tuple) and isinstance(value, TupleValue):
                _unpack_tensor_tuple(list(assign_target.elts), value.items, env, context)
            elif isinstance(assign_target, ast.Tuple) and isinstance(value, ShapeTupleValue):
                for t_elt, dim in zip(assign_target.elts, value.dims, strict=False):
                    if isinstance(t_elt, ast.Name):
                        env[t_elt.id] = dim
            else:
                _bind_target(assign_target, value, env, context)
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
        declared: TensorValue | None = None
        try:
            declared = parse_tensor_annotation(statement.annotation, aliases)
        except AnnotationParseError as error:
            context.error(statement, "TSF1001", error.message)
        _maybe_hover_alias_reference(statement.annotation, aliases, context)
        value = (
            _eval_expr(statement.value, env, context, module_specs)
            if statement.value is not None
            else None
        )
        if declared is not None:
            if isinstance(value, TensorValue) and _shapes_definitely_mismatch(
                declared.shape, value.shape
            ):
                context.error(
                    statement,
                    "TSF1011",
                    f"Assigned shape {value.shape} does not match declared {declared.shape}.",
                )
            _bind_target(statement.target, declared, env, context)
            return
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
        _analyze_if(statement, env, context, module_specs, aliases)
        return


def _analyze_if(
    node: ast.If,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
    aliases: dict[str, TensorValue],
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
    pre_aliases = dict(aliases)
    env_then: dict[str, Value] = dict(env)
    aliases_then: dict[str, TensorValue] = dict(aliases)
    for stmt in node.body:
        _analyze_statement(stmt, env_then, context, module_specs, aliases_then)
    if node.orelse:
        env_else: dict[str, Value] = dict(pre_env)
        aliases_else: dict[str, TensorValue] = dict(pre_aliases)
        for stmt in node.orelse:
            _analyze_statement(stmt, env_else, context, module_specs, aliases_else)
        _merge_envs(env, pre_env, env_then, env_else)
        _merge_aliases(aliases, pre_aliases, aliases_then, aliases_else)
    else:
        # No else: take the ``if`` body environment (pragmatically useful).
        env.update(env_then)
        aliases.clear()
        aliases.update(aliases_then)


def _merge_aliases(
    aliases: dict[str, TensorValue],
    pre_aliases: dict[str, TensorValue],
    aliases_then: dict[str, TensorValue],
    aliases_else: dict[str, TensorValue],
) -> None:
    """Merge branch-local alias scopes back into the current function scope."""
    all_keys = set(aliases_then) | set(aliases_else)
    aliases.clear()
    for key in all_keys:
        then_val = aliases_then.get(key)
        else_val = aliases_else.get(key)
        if then_val is not None and else_val is not None:
            if str(then_val.shape) == str(else_val.shape):
                aliases[key] = then_val
            elif key in pre_aliases:
                aliases[key] = pre_aliases[key]
        elif key in pre_aliases:
            aliases[key] = pre_aliases[key]


def _alias_target_node(statement: ast.stmt) -> ast.Name | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            return target
        return None
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        return statement.target
    alias_name = _type_alias_name_node(statement)
    if alias_name is not None:
        return alias_name
    return None


def _type_alias_name_node(statement: ast.stmt) -> ast.Name | None:
    type_alias_cls = getattr(ast, "TypeAlias", None)
    if type_alias_cls is None or not isinstance(statement, type_alias_cls):
        return None
    name = getattr(statement, "name", None)
    return name if isinstance(name, ast.Name) else None


def _maybe_hover_alias_reference(
    annotation: ast.AST,
    aliases: dict[str, TensorValue],
    context: ModuleContext,
) -> None:
    if isinstance(annotation, ast.Name):
        alias_value = aliases.get(annotation.id)
        if alias_value is not None:
            context.hover_alias(annotation.id, annotation, alias_value)


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
            LSTMSpec,
            CustomModuleSpec,
            RepeatSpec,
            SequentialSpec,
            TensorTupleValue,
            TupleValue,
        ),
    ):
        env[target.id] = value
        if isinstance(value, TensorValue):
            context.hover(target.id, target, value)


def _emit_matmul_mismatch(
    context: ModuleContext,
    node: ast.AST,
    op_label: str,
    left: TensorValue,
    right: TensorValue,
) -> None:
    """Emit a TSF1003 diagnostic for matmul/bmm-family shape mismatches."""
    context.shape_error(
        node,
        "TSF1003",
        f"Incompatible {op_label} shapes",
        expected=(
            "last dim of left to equal second-to-last dim of right"
            " (with broadcast-compatible batch dims)"
        ),
        actual=f"left={left.shape}, right={right.shape}",
        hint=(
            "transpose one operand with .transpose(-2, -1), or adjust an upstream"
            " layer so the inner dimensions agree"
        ),
    )


def _emit_mm_mismatch(
    context: ModuleContext,
    node: ast.AST,
    left: TensorValue,
    right: TensorValue,
) -> None:
    """Emit a TSF1003 diagnostic for torch.mm / Tensor.mm shape mismatches."""
    context.shape_error(
        node,
        "TSF1003",
        "Incompatible mm shapes",
        expected="two rank-2 tensors (M, N) and (N, K)",
        actual=f"left={left.shape}, right={right.shape}",
        hint=(
            "mm requires strict 2D tensors with matching inner dim;"
            " use matmul for broadcasting or reshape the operands"
        ),
    )


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
            tensor = context.self_tensors.get(node.attr)
            if tensor is not None:
                context.hover(node.attr, node, tensor)
                return tensor
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
        if isinstance(base, TupleValue):
            idx = int_from_ast(node.slice)
            if idx is not None:
                norm = normalize_index(idx, len(base.items))
                if norm is not None:
                    return base.items[norm]
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
                _emit_matmul_mismatch(context, node, "@ (matmul)", left, right)
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
) -> TensorValue | TensorTupleValue | TupleValue | None:
    """Apply a single module spec to an input tensor and return the output."""
    if isinstance(spec, LinearSpec):
        if spec.in_features is not None and tensor.rank > 0:
            last = tensor.shape.dims[-1]
            if isinstance(last, (SymbolicDim, ExpressionDim)):
                context.error(
                    node,
                    "TSF1012",
                    f"Symbolic dim '{render_dim(last)}' used where nn.Linear expects"
                    f" in_features={spec.in_features} — consider replacing with"
                    f" {spec.in_features} in your Shape annotation.",
                    severity="warning",
                )
        result = infer_linear(spec, tensor)
        if result is None:
            if tensor.rank == 0:
                context.shape_error(
                    node,
                    "TSF1007",
                    "nn.Linear input shape mismatch",
                    expected="rank ≥ 1",
                    actual="rank-0 tensor (scalar)",
                    hint="nn.Linear requires a tensor with at least one dimension",
                )
            else:
                # infer_linear only returns None with rank>=1 when in_features
                # is a literal int and the last dim is a ConstantDim that
                # differs. Both are therefore concrete here.
                last = tensor.shape.dims[-1]
                context.shape_error(
                    node,
                    "TSF1007",
                    "nn.Linear input shape mismatch",
                    expected=f"last dim = {spec.in_features}",
                    actual=f"{tensor.shape} (last dim = {render_dim(last)})",
                    hint=(
                        f"change nn.Linear(in_features=...) to {render_dim(last)},"
                        f" or reshape the input so its last dim equals {spec.in_features}"
                    ),
                )
        return result
    if isinstance(spec, Conv2dSpec):
        if spec.in_channels is not None and tensor.rank == 4:
            ch = tensor.shape.dims[1]
            if isinstance(ch, (SymbolicDim, ExpressionDim)):
                context.error(
                    node,
                    "TSF1012",
                    f"Symbolic dim '{render_dim(ch)}' used where nn.Conv2d expects"
                    f" in_channels={spec.in_channels} — consider replacing with"
                    f" {spec.in_channels} in your Shape annotation.",
                    severity="warning",
                )
        result = infer_conv2d(spec, tensor)
        if result is None:
            if tensor.rank != 4:
                context.shape_error(
                    node,
                    "TSF1007",
                    "nn.Conv2d input shape mismatch",
                    expected="rank-4 tensor (N, C, H, W)",
                    actual=f"rank-{tensor.rank} tensor {tensor.shape}",
                    hint="nn.Conv2d requires a 4D input; add batch / channel dims if missing",
                )
            else:
                # With rank==4, infer_conv2d only returns None when
                # in_channels is a literal int and the channel dim is a
                # ConstantDim that differs. Both are concrete here.
                ch = tensor.shape.dims[1]
                context.shape_error(
                    node,
                    "TSF1007",
                    "nn.Conv2d input shape mismatch",
                    expected=f"channels dim = {spec.in_channels}",
                    actual=f"{tensor.shape} (channels dim = {render_dim(ch)})",
                    hint=(
                        f"change nn.Conv2d(in_channels=...) to {render_dim(ch)},"
                        f" or reshape the input so dim 1 equals {spec.in_channels}"
                    ),
                )
        return result
    if isinstance(spec, PassthroughSpec):
        return tensor
    if isinstance(spec, EmbeddingSpec):
        return infer_embedding(spec, tensor)
    if isinstance(spec, Pool2dSpec):
        result = infer_pool2d(spec, tensor)
        if result is None:
            context.shape_error(
                node,
                "TSF1007",
                "nn.MaxPool2d/AvgPool2d input shape mismatch",
                expected="rank-4 tensor (N, C, H, W)",
                actual=f"rank-{tensor.rank} tensor {tensor.shape}",
                hint="2D pooling layers require a 4D input; add batch / channel dims if missing",
            )
        return result
    if isinstance(spec, CustomModuleSpec):
        if spec.input_shape is None or spec.return_shape is None:
            return None
        if spec.input_shape.rank != tensor.rank:
            return None
        for declared_dim, actual_dim in zip(
            spec.input_shape.shape.dims, tensor.shape.dims, strict=True
        ):
            if (
                isinstance(declared_dim, ConstantDim)
                and isinstance(actual_dim, ConstantDim)
                and declared_dim.value != actual_dim.value
            ):
                return None
        mapping = unify_dims(spec.input_shape.shape.dims, tensor.shape.dims)
        return TensorValue(apply_substitution(spec.return_shape.shape, mapping))
    if isinstance(spec, RepeatSpec):
        current = tensor
        if isinstance(spec.count, int):
            total = max(spec.count, spec.min_count)
            for _ in range(total):
                out = _apply_module_spec(spec.spec, current, node, context, module_specs)
                if not isinstance(out, TensorValue):
                    return None
                current = out
            return current

        for _ in range(spec.min_count):
            out = _apply_module_spec(spec.spec, current, node, context, module_specs)
            if not isinstance(out, TensorValue):
                return None
            current = out

        probe = _apply_module_spec(spec.spec, current, node, context, module_specs)
        if isinstance(probe, TensorValue) and str(probe.shape) == str(current.shape):
            return current
        return None
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
    if isinstance(spec, LSTMSpec):
        if spec.input_size is not None and tensor.rank == 3:
            last = tensor.shape.dims[-1]
            if isinstance(last, (SymbolicDim, ExpressionDim)):
                context.error(
                    node,
                    "TSF1012",
                    f"Symbolic dim '{render_dim(last)}' used where nn.LSTM expects"
                    f" input_size={spec.input_size} — consider replacing with"
                    f" {spec.input_size} in your Shape annotation.",
                    severity="warning",
                )
        lstm_out = infer_lstm(spec, tensor)
        if lstm_out is None:
            if tensor.rank != 3:
                expected_layout = (
                    "rank-3 tensor (N, L, input_size)"
                    if spec.batch_first
                    else "rank-3 tensor (L, N, input_size)"
                )
                context.shape_error(
                    node,
                    "TSF1007",
                    "nn.LSTM input shape mismatch",
                    expected=expected_layout,
                    actual=f"rank-{tensor.rank} tensor {tensor.shape}",
                    hint="nn.LSTM requires a 3D input; check batch_first to confirm dim order",
                )
            else:
                # With rank==3, infer_lstm only returns None when input_size
                # is a literal int and the last dim is a ConstantDim that
                # differs. Both are concrete here.
                last = tensor.shape.dims[-1]
                context.shape_error(
                    node,
                    "TSF1007",
                    "nn.LSTM input shape mismatch",
                    expected=f"last dim = {spec.input_size}",
                    actual=f"{tensor.shape} (last dim = {render_dim(last)})",
                    hint=(
                        f"change nn.LSTM(input_size=...) to {render_dim(last)},"
                        f" or reshape the input so its last dim equals {spec.input_size}"
                    ),
                )
        return lstm_out
    return None


def _module_spec_from_value(value: Value | Dim | None) -> ModuleSpec | None:
    if isinstance(value, _MODULE_SPEC_TYPES):
        return value
    return None


def _call_has_tensor_arg(
    args: Sequence[ast.expr],
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> bool:
    for arg_node in args:
        arg_val = _eval_expr(arg_node, env, context, module_specs)
        if isinstance(arg_val, TensorValue):
            return True
    return False


def _eval_named_module_alias_call(
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> Value | None:
    if not isinstance(node.func, ast.Name):
        return None
    spec = _module_spec_from_value(env.get(node.func.id))
    if spec is None:
        return None
    argument = _eval_expr(node.args[0], env, context, module_specs) if node.args else None
    if isinstance(argument, TensorValue):
        return _apply_module_spec(spec, argument, node, context, module_specs)
    return None


def _eval_signature_match(
    node: ast.Call,
    sig: FuncSig,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> TensorValue | None:
    if sig.return_shape is None:
        return None
    mapping: dict[str, Dim] = {}
    for param_tv, arg_node in zip(sig.param_shapes, node.args, strict=False):
        if param_tv is None:
            continue
        arg_val = _eval_expr(arg_node, env, context, module_specs)
        if isinstance(arg_val, TensorValue):
            sub = unify_dims(param_tv.shape.dims, arg_val.shape.dims)
            for sym_name, bound_dim in sub.items():
                if sym_name in mapping and render_dim(mapping[sym_name]) != render_dim(bound_dim):
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


def _eval_signature_call(
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> TensorValue | None:
    if not isinstance(node.func, ast.Name):
        return None
    sig = context.func_sigs.get(node.func.id)
    if sig is None:
        return None
    return _eval_signature_match(node, sig, env, context, module_specs)


def _maybe_warn_unannotated_function_call(
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> None:
    if not isinstance(node.func, ast.Name):
        return
    if not context.in_annotated_function:
        return
    if node.func.id in _BUILTIN_NAMES or node.func.id in context.func_sigs:
        return
    if _module_spec_from_value(env.get(node.func.id)) is not None:
        return
    if _call_has_tensor_arg(node.args, env, context, module_specs):
        func_name = node.func.id
        context.error(
            node,
            "TSF2002",
            f"Call to unannotated function '{func_name}'"
            " — shape not tracked. Consider adding a Shape annotation.",
            severity="warning",
        )


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
            # Check for a submodule spec (nn.Linear, etc.)
            owner = module_specs.get(node.func.attr)
            # Check for a method signature (self.helper_method())
            method_sig = context.method_sigs.get(node.func.attr)
            if method_sig is not None:
                result = _eval_signature_match(node, method_sig, env, context, module_specs)
                if isinstance(result, TensorValue):
                    context.hover(node.func.attr, node, result)
                return result
        else:
            owner = _eval_expr(node.func.value, env, context, module_specs)
        if isinstance(owner, TensorValue):
            return _eval_tensor_method(owner, node, context, env, module_specs)
        spec = _module_spec_from_value(owner)
        if spec is not None:
            argument = _eval_expr(node.args[0], env, context, module_specs) if node.args else None
            if isinstance(argument, TensorValue):
                return _apply_module_spec(spec, argument, node, context, module_specs)
        # TSF2003: self.xxx(tensor) where xxx has no spec.
        if is_self_call and owner is None and context.in_annotated_function:
            if _call_has_tensor_arg(node.args, env, context, module_specs):
                attr = node.func.attr
                if attr not in context.method_sigs:
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
                    op_label = "bmm" if callee_name.endswith("bmm") else "matmul"
                    _emit_matmul_mismatch(context, node, op_label, left, right)
                return result
    if callee_name.endswith(".mm") or callee_name == "mm":
        if len(node.args) >= 2:
            left = _eval_expr(node.args[0], env, context, module_specs)
            right = _eval_expr(node.args[1], env, context, module_specs)
            if isinstance(left, TensorValue) and isinstance(right, TensorValue):
                result = infer_mm(left, right)
                if result is None:
                    _emit_mm_mismatch(context, node, left, right)
                return result
    if callee_name.endswith(".movedim") or callee_name == "movedim":
        if len(node.args) >= 3:
            tensor = _eval_expr(node.args[0], env, context, module_specs)
            if isinstance(tensor, TensorValue):
                src = _int_or_tuple(node.args[1])
                dst = _int_or_tuple(node.args[2])
                if src is not None and dst is not None:
                    result = infer_movedim(tensor, src, dst)
                    if result is None:
                        context.error(node, "TSF1008", "Invalid movedim dimensions.")
                    return result
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
                    shapes_str = ", ".join(str(t.shape) for t in tensor_list)
                    context.shape_error(
                        node,
                        "TSF1003",
                        "Incompatible einsum shapes",
                        expected=f"shapes consistent with subscript '{subscript_str}'",
                        actual=f"tensors: {shapes_str}",
                        hint=(
                            "check that each letter in the subscript maps to a consistent "
                            "size across all operands"
                        ),
                    )
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
    module_alias_result = _eval_named_module_alias_call(node, env, context, module_specs)
    if module_alias_result is not None:
        return module_alias_result
    signature_result = _eval_signature_call(node, env, context, module_specs)
    if signature_result is not None:
        return signature_result
    _maybe_warn_unannotated_function_call(node, env, context, module_specs)
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
            if context.in_annotated_function:
                context.error(
                    node,
                    "TSF2001",
                    "Cannot resolve '.permute' order statically — shape not tracked.",
                    severity="warning",
                )
            return None
        result = infer_permute(tensor, tuple(item for item in order if item is not None))
        if result is None:
            context.error(node, "TSF1008", "Invalid permutation.")
        return result
    if name == "transpose" and len(node.args) == 2:
        first = int_from_ast(node.args[0])
        second = int_from_ast(node.args[1])
        if first is None or second is None:
            if context.in_annotated_function:
                context.error(
                    node,
                    "TSF2001",
                    "Cannot resolve '.transpose' dims statically — shape not tracked.",
                    severity="warning",
                )
            return None
        result = infer_transpose(tensor, first, second)
        if result is None:
            context.error(node, "TSF1008", "Invalid transpose dimensions.")
        return result
    if name == "flatten":
        start_dim = _positional_int(node.args, 0, 0)
        end_dim = _positional_int(node.args, 1, -1)
        if start_dim is None or end_dim is None:
            if context.in_annotated_function:
                context.error(
                    node,
                    "TSF2001",
                    "Cannot resolve '.flatten' dims statically — shape not tracked.",
                    severity="warning",
                )
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
            if context.in_annotated_function:
                context.error(
                    node,
                    "TSF2001",
                    "Cannot resolve '.unsqueeze' dim statically — shape not tracked.",
                    severity="warning",
                )
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
                _emit_matmul_mismatch(context, node, "matmul", tensor, right)
            return result
    if name == "mm" and node.args:
        right = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(right, TensorValue):
            result = infer_mm(tensor, right)
            if result is None:
                _emit_mm_mismatch(context, node, tensor, right)
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
                if chunk_result is None:
                    context.error(node, "TSF1008", "Invalid chunk dimension.")
                return chunk_result
    if name == "split" and node.args:
        split_result = _split_from_call(tensor, node)
        if split_result is not None:
            return split_result
    if name == "movedim" and len(node.args) >= 2:
        src = _int_or_tuple(node.args[0])
        dst = _int_or_tuple(node.args[1])
        if src is not None and dst is not None:
            result = infer_movedim(tensor, src, dst)
            if result is None:
                context.error(node, "TSF1008", "Invalid movedim dimensions.")
            return result
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


def _dim_binop(op: ast.operator, left: Dim | int, right: Dim | int) -> Dim | None:
    """Apply an arithmetic operator to two dims, returning a combined Dim."""
    ld = ConstantDim(left) if isinstance(left, int) else left
    rd = ConstantDim(right) if isinstance(right, int) else right
    if isinstance(op, ast.Mult):
        return product_dim((ld, rd))
    if isinstance(op, ast.Add):
        return sum_dim((ld, rd))
    if isinstance(op, ast.Sub):
        if isinstance(ld, ConstantDim) and isinstance(rd, ConstantDim):
            return ConstantDim(ld.value - rd.value)
        return ExpressionDim(f"({render_dim(ld)} - {render_dim(rd)})")
    if isinstance(op, ast.FloorDiv):
        return quotient_dim((ld,), (rd,))
    return None


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
    # self.attr — look up scalar attributes stored from __init__ assignments.
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in context.self_scalars
    ):
        return context.self_scalars[node.attr]
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Mult, ast.Add, ast.Sub, ast.FloorDiv)
    ):
        left = _dim_from_expr(node.left, env, context, module_specs)
        right = _dim_from_expr(node.right, env, context, module_specs)
        if left is not None and right is not None:
            return _dim_binop(node.op, left, right)
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
            result_dims.append(product_dim((ConstantDim(repeat_val), d)))
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

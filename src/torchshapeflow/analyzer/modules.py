"""Class spec collection for the analyzer.

Walks ``ClassDef`` bodies and the ``__init__`` of each class to build the
``ModuleSpec`` map (``self.linear`` → ``LinearSpec``, etc.). Also handles
loop-built ``Sequential`` stacks and custom-module templates.
"""

from __future__ import annotations

import ast

from torchshapeflow.analysis_context import ModuleContext
from torchshapeflow.analyzer.constants import PASSTHROUGH_SUFFIXES
from torchshapeflow.ast_helpers import (
    alias_target_node,
    int_pair,
    is_first_iteration_test,
    keyword_or_default,
    pool_stride,
)
from torchshapeflow.index import (
    CustomModuleTemplate,
    apply_substitution,
    extract_alias_binding,
)
from torchshapeflow.model import (
    Conv2dSpec,
    CustomModuleSpec,
    Dim,
    EmbeddingSpec,
    LinearSpec,
    LSTMSpec,
    ModuleSpec,
    MultiheadAttentionSpec,
    PassthroughSpec,
    Pool2dSpec,
    RepeatSpec,
    SequentialSpec,
    SymbolicDim,
    TensorValue,
    make_dim,
)
from torchshapeflow.rules.common import int_from_ast, qualified_name


def collect_class_specs(
    module: ast.Module,
    context: ModuleContext,
    custom_module_templates: dict[str, CustomModuleTemplate],
) -> tuple[
    dict[str, dict[str, ModuleSpec]],
    dict[str, dict[str, Dim]],
    dict[str, dict[str, TensorValue]],
]:
    # Late import: functions.py depends on statements.py / expressions.py via
    # the walker; statements/expressions still live in analyzer/__init__.py
    # for this stage of the split. Pulling at call-time avoids the cycle.
    from torchshapeflow.analyzer.functions import (
        collect_function_annotations as _collect_function_annotations,
    )
    from torchshapeflow.analyzer.functions import (
        init_tensor_from_expr as _init_tensor_from_expr,
    )
    from torchshapeflow.analyzer.functions import (
        is_register_buffer_call as _is_register_buffer_call,
    )
    from torchshapeflow.analyzer.functions import (
        register_buffer_binding as _register_buffer_binding,
    )

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
                positive_params = collect_positive_scalar_params(child)
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
                            spec = parse_module_spec(
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
                        sequence_binding = loop_sequence_binding(
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


def emit_module_alias_hovers(module: ast.Module, context: ModuleContext) -> None:
    for statement in module.body:
        alias = extract_alias_binding(statement)
        if alias is None:
            continue
        alias_name, alias_node = alias
        tensor = context.aliases.get(alias_name)
        target = alias_target_node(statement)
        if tensor is not None and target is not None:
            context.hover_alias(alias_name, target, tensor)


def parse_module_spec(
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
        kernel_size = int_pair(node.args[2])
        stride = int_pair(keyword_or_default(node, "stride"), default=(1, 1))
        padding = int_pair(keyword_or_default(node, "padding"), default=(0, 0))
        dilation = int_pair(keyword_or_default(node, "dilation"), default=(1, 1))
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
        kernel_size = int_pair(node.args[0])
        if kernel_size is not None:
            stride = pool_stride(node, kernel_size)
            padding = int_pair(keyword_or_default(node, "padding"), default=(0, 0))
            dilation = int_pair(keyword_or_default(node, "dilation"), default=(1, 1))
            if stride is not None and padding is not None and dilation is not None:
                return Pool2dSpec(
                    kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
                )
    if name.endswith("AvgPool2d") and node.args:
        kernel_size = int_pair(node.args[0])
        if kernel_size is not None:
            stride = pool_stride(node, kernel_size)
            padding = int_pair(keyword_or_default(node, "padding"), default=(0, 0))
            if stride is not None and padding is not None:
                return Pool2dSpec(
                    kernel_size=kernel_size, stride=stride, padding=padding, dilation=(1, 1)
                )
    if short_name in PASSTHROUGH_SUFFIXES:
        return PassthroughSpec()
    if short_name == "Sequential":
        sub_specs: list[ModuleSpec] = []
        for arg in node.args:
            sub_node = arg.value if isinstance(arg, ast.Starred) else arg
            sub = parse_module_spec(sub_node, sequence_specs, custom_module_templates)
            if sub is None:
                return None
            sub_specs.append(sub)
        return SequentialSpec(specs=tuple(sub_specs))
    if short_name == "ModuleList" and len(node.args) == 1:
        return parse_module_spec(node.args[0], sequence_specs, custom_module_templates)
    if custom_module_templates is not None:
        template = custom_module_templates.get(short_name)
        if template is not None:
            return bind_custom_module_template(node, template)
    if short_name == "MultiheadAttention" and len(node.args) >= 2:
        embed_dim = _int(node.args[0])
        num_heads = _int(node.args[1])
        batch_first_node = keyword_or_default(node, "batch_first")
        batch_first = False
        if isinstance(batch_first_node, ast.Constant) and isinstance(batch_first_node.value, bool):
            batch_first = batch_first_node.value
        if embed_dim is not None and num_heads is not None:
            return MultiheadAttentionSpec(
                embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first
            )
    if short_name == "LSTM" and (len(node.args) >= 2 or keyword_or_default(node, "hidden_size")):
        input_size_node = (
            node.args[0] if len(node.args) >= 1 else keyword_or_default(node, "input_size")
        )
        hidden_size_node = (
            node.args[1] if len(node.args) >= 2 else keyword_or_default(node, "hidden_size")
        )
        if hidden_size_node is None:
            return None
        input_size = _int(input_size_node) if input_size_node is not None else None
        hidden_size = _sym(hidden_size_node)
        proj_size: int | str | None = None
        proj_size_arg = keyword_or_default(node, "proj_size")
        if proj_size_arg is not None:
            proj_size = _sym(proj_size_arg)
        num_layers: int | str = 1
        num_layers_arg = (
            node.args[2] if len(node.args) >= 3 else keyword_or_default(node, "num_layers")
        )
        if num_layers_arg is not None:
            parsed = _sym(num_layers_arg)
            if parsed is not None:
                num_layers = parsed
        batch_first_node = keyword_or_default(node, "batch_first")
        batch_first = False
        if isinstance(batch_first_node, ast.Constant) and isinstance(batch_first_node.value, bool):
            batch_first = batch_first_node.value
        bidirectional_node = keyword_or_default(node, "bidirectional")
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


def bind_custom_module_template(
    call: ast.Call,
    template: CustomModuleTemplate,
) -> CustomModuleSpec:
    mapping: dict[str, Dim] = {}
    for index, param_name in enumerate(template.init_param_names):
        arg_node: ast.AST | None
        if index < len(call.args):
            arg_node = call.args[index]
        else:
            arg_node = keyword_or_default(call, param_name)
        dim_value = dim_symbol_from_ast(arg_node) if arg_node is not None else None
        if dim_value is not None:
            mapping[param_name] = make_dim(dim_value)

    input_shape = substitute_tensor_value(template.input_shape, mapping)
    return_shape = substitute_tensor_value(template.return_shape, mapping)
    return CustomModuleSpec(input_shape=input_shape, return_shape=return_shape)


def substitute_tensor_value(
    tensor: TensorValue | None, mapping: dict[str, Dim]
) -> TensorValue | None:
    if tensor is None:
        return None
    return TensorValue(apply_substitution(tensor.shape, mapping), origin=tensor.origin)


def dim_symbol_from_ast(node: ast.AST | None) -> int | str | None:
    if node is None:
        return None
    val = int_from_ast(node)
    if val is not None:
        return val
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Add, ast.Sub)):
        left = dim_symbol_from_ast(node.left)
        right = dim_symbol_from_ast(node.right)
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


def collect_positive_scalar_params(init_func: ast.FunctionDef) -> set[str]:
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


def loop_sequence_binding(
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

    first_node = resolve_loop_expr(append_arg, local_bindings, loop_var, is_first=True)
    rest_node = resolve_loop_expr(append_arg, local_bindings, loop_var, is_first=False)
    first_spec = parse_module_spec(first_node, sequence_specs, custom_module_templates)
    rest_spec = parse_module_spec(rest_node, sequence_specs, custom_module_templates)
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


def resolve_loop_expr(
    node: ast.expr,
    bindings: dict[str, ast.expr],
    loop_var: str,
    *,
    is_first: bool,
) -> ast.expr:
    if isinstance(node, ast.Name) and node.id in bindings:
        return resolve_loop_expr(bindings[node.id], bindings, loop_var, is_first=is_first)
    if isinstance(node, ast.IfExp) and is_first_iteration_test(node.test, loop_var):
        branch = node.body if is_first else node.orelse
        return resolve_loop_expr(branch, bindings, loop_var, is_first=is_first)
    if isinstance(node, ast.Call):
        return ast.Call(
            func=resolve_loop_expr(node.func, bindings, loop_var, is_first=is_first),
            args=[
                resolve_loop_expr(arg, bindings, loop_var, is_first=is_first) for arg in node.args
            ],
            keywords=[
                ast.keyword(
                    arg=kw.arg,
                    value=resolve_loop_expr(kw.value, bindings, loop_var, is_first=is_first),
                )
                for kw in node.keywords
            ],
        )
    if isinstance(node, ast.BinOp):
        return ast.BinOp(
            left=resolve_loop_expr(node.left, bindings, loop_var, is_first=is_first),
            op=node.op,
            right=resolve_loop_expr(node.right, bindings, loop_var, is_first=is_first),
        )
    if isinstance(node, ast.Attribute):
        return ast.Attribute(
            value=resolve_loop_expr(node.value, bindings, loop_var, is_first=is_first),
            attr=node.attr,
            ctx=node.ctx,
        )
    return node

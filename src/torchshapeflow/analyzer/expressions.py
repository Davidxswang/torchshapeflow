"""Top-level expression evaluator: ``_eval_expr`` plus ``_apply_module_spec``
and the small signature-call / module-alias / unannotated-function helpers.

``_eval_expr`` is the central dispatcher used across the analyzer; ``_eval_call``
and ``_eval_tensor_method`` (in their own modules) call back into ``_eval_expr``
for sub-expressions, and ``_eval_expr`` calls into ``_eval_call`` for ``ast.Call``
nodes — so the latter is pulled in via a late import inside this function to
break the cycle.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

from torchshapeflow.analysis_context import ModuleContext
from torchshapeflow.analyzer.constants import BUILTIN_NAMES, MODULE_SPEC_TYPES
from torchshapeflow.analyzer.statements import emit_matmul_mismatch
from torchshapeflow.arithmetic import broadcast_has_uncertain_dims, normalize_index
from torchshapeflow.index import FuncSig, apply_substitution, unify_dims
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
    render_dim,
)
from torchshapeflow.rules import (
    infer_binary_broadcast,
    infer_conv2d,
    infer_embedding,
    infer_linear,
    infer_lstm,
    infer_matmul,
    infer_pool2d,
    infer_subscript,
)
from torchshapeflow.rules.common import int_from_ast


def eval_expr(
    node: ast.AST,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> Value | Dim | None:
    # Late import: calls.py is the dispatcher for ast.Call nodes; importing it
    # at module load would form a cycle.
    from torchshapeflow.analyzer.calls import eval_call

    if isinstance(node, ast.Name):
        value = env.get(node.id)
        if isinstance(value, TensorValue):
            context.hover(node.id, node, value)
        return value
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return IntegerValue(node.value)
    if isinstance(node, ast.Attribute):
        base = eval_expr(node.value, env, context, module_specs)
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
        base = eval_expr(node.value, env, context, module_specs)
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
        left = eval_expr(node.left, env, context, module_specs)
        right = eval_expr(node.right, env, context, module_specs)
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
        left = eval_expr(node.left, env, context, module_specs)
        right = eval_expr(node.right, env, context, module_specs)
        if isinstance(left, TensorValue) and isinstance(right, TensorValue):
            result = infer_matmul(left, right)
            if result is None:
                emit_matmul_mismatch(context, node, "@ (matmul)", left, right)
            return result
        return None
    if isinstance(node, ast.Call):
        return eval_call(node, env, context, module_specs)
    if isinstance(node, ast.Tuple):
        return None
    return None


def apply_module_spec(
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
                out = apply_module_spec(spec.spec, current, node, context, module_specs)
                if not isinstance(out, TensorValue):
                    return None
                current = out
            return current

        for _ in range(spec.min_count):
            out = apply_module_spec(spec.spec, current, node, context, module_specs)
            if not isinstance(out, TensorValue):
                return None
            current = out

        probe = apply_module_spec(spec.spec, current, node, context, module_specs)
        if isinstance(probe, TensorValue) and str(probe.shape) == str(current.shape):
            return current
        return None
    if isinstance(spec, SequentialSpec):
        current = tensor
        for sub in spec.specs:
            out = apply_module_spec(sub, current, node, context, module_specs)
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


def module_spec_from_value(value: Value | Dim | None) -> ModuleSpec | None:
    if isinstance(value, MODULE_SPEC_TYPES):
        return value
    return None


def call_has_tensor_arg(
    args: Sequence[ast.expr],
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> bool:
    for arg_node in args:
        arg_val = eval_expr(arg_node, env, context, module_specs)
        if isinstance(arg_val, TensorValue):
            return True
    return False


def eval_named_module_alias_call(
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> Value | None:
    if not isinstance(node.func, ast.Name):
        return None
    spec = module_spec_from_value(env.get(node.func.id))
    if spec is None:
        return None
    argument = eval_expr(node.args[0], env, context, module_specs) if node.args else None
    if isinstance(argument, TensorValue):
        return apply_module_spec(spec, argument, node, context, module_specs)
    return None


def eval_signature_match(
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
        arg_val = eval_expr(arg_node, env, context, module_specs)
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


def eval_signature_call(
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
    return eval_signature_match(node, sig, env, context, module_specs)


def maybe_warn_unannotated_function_call(
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> None:
    if not isinstance(node.func, ast.Name):
        return
    if not context.in_annotated_function:
        return
    if node.func.id in BUILTIN_NAMES or node.func.id in context.func_sigs:
        return
    if module_spec_from_value(env.get(node.func.id)) is not None:
        return
    if call_has_tensor_arg(node.args, env, context, module_specs):
        func_name = node.func.id
        context.error(
            node,
            "TSF2002",
            f"Call to unannotated function '{func_name}'"
            " — shape not tracked. Consider adding a Shape annotation.",
            severity="warning",
        )

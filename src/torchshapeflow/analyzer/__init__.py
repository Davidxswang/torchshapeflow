"""Analyzer subpackage entry: re-exports the public API plus the residual
expression-evaluation walker (``_eval_expr`` / ``_eval_call`` /
``_eval_tensor_method``). The walker still lives here because all three are
mutually recursive; splitting them is a follow-up.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

from torchshapeflow.analysis_context import ModuleContext
from torchshapeflow.analyzer.constants import (
    BUILTIN_NAMES as _BUILTIN_NAMES,
)
from torchshapeflow.analyzer.constants import (
    FUNCTIONAL_PASSTHROUGH as _FUNCTIONAL_PASSTHROUGH,
)
from torchshapeflow.analyzer.constants import (
    LIKE_OPS as _LIKE_OPS,
)
from torchshapeflow.analyzer.constants import (
    MODULE_SPEC_TYPES as _MODULE_SPEC_TYPES,
)
from torchshapeflow.analyzer.constants import (
    NON_TENSOR_METHODS as _NON_TENSOR_METHODS,
)
from torchshapeflow.analyzer.constants import (
    PASSTHROUGH_METHODS as _PASSTHROUGH_METHODS,
)
from torchshapeflow.analyzer.constants import (
    REDUCTION_OPS as _REDUCTION_OPS,
)
from torchshapeflow.analyzer.constants import (
    TENSOR_CONSTRUCTORS as _TENSOR_CONSTRUCTORS,
)
from torchshapeflow.analyzer.entry import analyze_path, analyze_source
from torchshapeflow.analyzer.statements import emit_matmul_mismatch, emit_mm_mismatch
from torchshapeflow.arithmetic import broadcast_has_uncertain_dims, normalize_index
from torchshapeflow.ast_helpers import (
    arange_length,
    dim_binop,
    infer_repeat_call,
    int_or_tuple,
    keyword_int,
    keyword_or_default,
    positional_int,
    reduction_dim,
    reduction_keepdim,
    split_from_call,
)
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
    make_dim,
    render_dim,
)
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

__all__ = ["analyze_path", "analyze_source"]


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
                emit_matmul_mismatch(context, node, "@ (matmul)", left, right)
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
        dim = keyword_int(node, "dim", 0)
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
        dim = keyword_int(node, "dim", 0)
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
                    emit_matmul_mismatch(context, node, op_label, left, right)
                return result
    if callee_name.endswith(".mm") or callee_name == "mm":
        if len(node.args) >= 2:
            left = _eval_expr(node.args[0], env, context, module_specs)
            right = _eval_expr(node.args[1], env, context, module_specs)
            if isinstance(left, TensorValue) and isinstance(right, TensorValue):
                result = infer_mm(left, right)
                if result is None:
                    emit_mm_mismatch(context, node, left, right)
                return result
    if callee_name.endswith(".movedim") or callee_name == "movedim":
        if len(node.args) >= 3:
            tensor = _eval_expr(node.args[0], env, context, module_specs)
            if isinstance(tensor, TensorValue):
                src = int_or_tuple(node.args[1])
                dst = int_or_tuple(node.args[2])
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
            rdim = reduction_dim(node, arg_offset=1)
            keepdim = reduction_keepdim(node, positional_index=2)
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
                nc_node = keyword_or_default(node, "num_classes")
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
                k_node = keyword_or_default(node, "k")
                if k_node is not None:
                    k_val = int_from_ast(k_node)
            dim_val = positional_int(node.args, 2, -1)
            if dim_val is None:
                dim_val = keyword_int(node, "dim", -1)
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
            offset_val = positional_int(node.args, 1, None)
            if offset_val is None:
                offset_val = keyword_int(node, "offset", 0)
            if offset_val is None:
                offset_val = 0
            dim1_val = positional_int(node.args, 2, None)
            if dim1_val is None:
                dim1_val = keyword_int(node, "dim1", 0)
            if dim1_val is None:
                dim1_val = 0
            dim2_val = positional_int(node.args, 3, None)
            if dim2_val is None:
                dim2_val = keyword_int(node, "dim2", 1)
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
        arange_len = arange_length(node)
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
            dim = positional_int(node.args, 2, None)
            if dim is None:
                dim = keyword_int(node, "dim", 0)
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
        start_dim = positional_int(node.args, 0, 0)
        end_dim = positional_int(node.args, 1, -1)
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
        dim = positional_int(node.args, 0, None)
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
        dim = positional_int(node.args, 0, None)
        return infer_size(tensor, dim)
    if name == "matmul" and node.args:
        right = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(right, TensorValue):
            result = infer_matmul(tensor, right)
            if result is None:
                emit_matmul_mismatch(context, node, "matmul", tensor, right)
            return result
    if name == "mm" and node.args:
        right = _eval_expr(node.args[0], env, context, module_specs)
        if isinstance(right, TensorValue):
            result = infer_mm(tensor, right)
            if result is None:
                emit_mm_mismatch(context, node, tensor, right)
            return result
    if name in _REDUCTION_OPS:
        rdim = reduction_dim(node, arg_offset=0)
        keepdim = reduction_keepdim(node, positional_index=1)
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
        return infer_repeat_call(tensor, node)
    if name == "chunk" and node.args:
        n = int_from_ast(node.args[0])
        if n is not None:
            dim = positional_int(node.args, 1, None)
            if dim is None:
                dim = keyword_int(node, "dim", 0)
            if dim is not None:
                chunk_result = infer_chunk(tensor, n, dim)
                if chunk_result is None:
                    context.error(node, "TSF1008", "Invalid chunk dimension.")
                return chunk_result
    if name == "split" and node.args:
        split_result = split_from_call(tensor, node)
        if split_result is not None:
            return split_result
    if name == "movedim" and len(node.args) >= 2:
        src = int_or_tuple(node.args[0])
        dst = int_or_tuple(node.args[1])
        if src is not None and dst is not None:
            result = infer_movedim(tensor, src, dst)
            if result is None:
                context.error(node, "TSF1008", "Invalid movedim dimensions.")
            return result
    if name == "diagonal":
        offset_val = positional_int(node.args, 0, 0)
        if offset_val is None:
            offset_val = 0
        dim1_val = positional_int(node.args, 1, None)
        if dim1_val is None:
            dim1_val = keyword_int(node, "dim1", 0)
        if dim1_val is None:
            dim1_val = 0
        dim2_val = positional_int(node.args, 2, None)
        if dim2_val is None:
            dim2_val = keyword_int(node, "dim2", 1)
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
            k_node = keyword_or_default(node, "k")
            if k_node is not None:
                k_val = int_from_ast(k_node)
        dim_val = positional_int(node.args, 1, None)
        if dim_val is None:
            dim_val = keyword_int(node, "dim", -1)
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
            return dim_binop(node.op, left, right)
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
        size_node = keyword_or_default(node, "size")
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
    scale_node = keyword_or_default(node, "scale_factor")
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

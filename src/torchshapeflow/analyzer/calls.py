"""Function-call dispatcher: ``eval_call`` plus the size/constructor/interpolate
helpers used to evaluate ``ast.Call`` nodes.

Cross-module recursion: ``eval_call`` recurses on argument expressions via
``eval_expr`` (top-level import from ``expressions``) and dispatches tensor
methods via ``eval_tensor_method`` (late import from ``tensor_methods`` to
break the cycle, since ``tensor_methods`` imports ``_size_to_dim`` /
``_reshape_from_args`` from this module).
"""

from __future__ import annotations

import ast
from collections.abc import Sequence

from torchshapeflow.analysis_context import ModuleContext
from torchshapeflow.analyzer.constants import (
    FUNCTIONAL_PASSTHROUGH,
    LIKE_OPS,
    REDUCTION_OPS,
    TENSOR_CONSTRUCTORS,
)
from torchshapeflow.analyzer.expressions import (
    apply_module_spec,
    call_has_tensor_arg,
    eval_expr,
    eval_named_module_alias_call,
    eval_signature_call,
    eval_signature_match,
    maybe_warn_unannotated_function_call,
    module_spec_from_value,
)
from torchshapeflow.analyzer.statements import emit_matmul_mismatch, emit_mm_mismatch
from torchshapeflow.ast_helpers import (
    arange_length,
    dim_binop,
    int_or_tuple,
    keyword_int,
    keyword_or_default,
    positional_int,
    reduction_dim,
    reduction_keepdim,
)
from torchshapeflow.model import (
    ConstantDim,
    Dim,
    ExpressionDim,
    IntegerValue,
    ModuleSpec,
    ShapeTupleValue,
    SymbolicDim,
    TensorShape,
    TensorValue,
    UnknownDim,
    Value,
    make_dim,
)
from torchshapeflow.rules import (
    infer_cat,
    infer_diagonal,
    infer_einsum,
    infer_interpolate,
    infer_matmul,
    infer_mm,
    infer_movedim,
    infer_one_hot,
    infer_reduction,
    infer_reshape,
    infer_split,
    infer_stack,
    infer_topk,
)
from torchshapeflow.rules.common import int_from_ast, qualified_name


def eval_call(
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> Value | None:
    # Late import: tensor_methods.py imports _size_to_dim / _reshape_from_args
    # from this module at load time, so importing eval_tensor_method top-level
    # would cycle.
    from torchshapeflow.analyzer.tensor_methods import eval_tensor_method

    callee_name = qualified_name(node.func)
    if isinstance(node.func, ast.Attribute):
        owner: Value | Dim | None
        is_self_call = isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
        if is_self_call:
            owner = module_specs.get(node.func.attr)
            method_sig = context.method_sigs.get(node.func.attr)
            if method_sig is not None:
                result = eval_signature_match(node, method_sig, env, context, module_specs)
                if isinstance(result, TensorValue):
                    context.hover(node.func.attr, node, result)
                return result
        else:
            owner = eval_expr(node.func.value, env, context, module_specs)
        if isinstance(owner, TensorValue):
            return eval_tensor_method(owner, node, context, env, module_specs)
        spec = module_spec_from_value(owner)
        if spec is not None:
            argument = eval_expr(node.args[0], env, context, module_specs) if node.args else None
            if isinstance(argument, TensorValue):
                return apply_module_spec(spec, argument, node, context, module_specs)
        if is_self_call and owner is None and context.in_annotated_function:
            if call_has_tensor_arg(node.args, env, context, module_specs):
                attr = node.func.attr
                if attr not in context.method_sigs:
                    context.error(
                        node,
                        "TSF2003",
                        f"No shape spec for 'self.{attr}' — shape not tracked.",
                        severity="warning",
                    )
    if callee_name.endswith("reshape") and len(node.args) >= 2:
        tensor = eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor, TensorValue):
            return reshape_from_args(tensor, node.args[1:], context, node, env, module_specs)
    if callee_name.endswith("cat") and node.args:
        values = tensor_sequence(node.args[0], env, context, module_specs)
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
        values = tensor_sequence(node.args[0], env, context, module_specs)
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
            left = eval_expr(node.args[0], env, context, module_specs)
            right = eval_expr(node.args[1], env, context, module_specs)
            if isinstance(left, TensorValue) and isinstance(right, TensorValue):
                result = infer_matmul(left, right)
                if result is None:
                    op_label = "bmm" if callee_name.endswith("bmm") else "matmul"
                    emit_matmul_mismatch(context, node, op_label, left, right)
                return result
    if callee_name.endswith(".mm") or callee_name == "mm":
        if len(node.args) >= 2:
            left = eval_expr(node.args[0], env, context, module_specs)
            right = eval_expr(node.args[1], env, context, module_specs)
            if isinstance(left, TensorValue) and isinstance(right, TensorValue):
                result = infer_mm(left, right)
                if result is None:
                    emit_mm_mismatch(context, node, left, right)
                return result
    if callee_name.endswith(".movedim") or callee_name == "movedim":
        if len(node.args) >= 3:
            tensor = eval_expr(node.args[0], env, context, module_specs)
            if isinstance(tensor, TensorValue):
                src = int_or_tuple(node.args[1])
                dst = int_or_tuple(node.args[2])
                if src is not None and dst is not None:
                    result = infer_movedim(tensor, src, dst)
                    if result is None:
                        context.error(node, "TSF1008", "Invalid movedim dimensions.")
                    return result
    if callee_name.endswith("einsum") and node.args:
        subscript_node = node.args[0]
        if isinstance(subscript_node, ast.Constant) and isinstance(subscript_node.value, str):
            subscript_str = subscript_node.value
            if len(node.args) == 2 and isinstance(node.args[1], (ast.List, ast.Tuple)):
                tensor_arg_nodes: list[ast.expr] = list(node.args[1].elts)
            else:
                tensor_arg_nodes = list(node.args[1:])
            tensor_vals = [eval_expr(a, env, context, module_specs) for a in tensor_arg_nodes]
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
    if callee_name.endswith("interpolate") and node.args:
        tensor = eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor, TensorValue):
            n_spatial = tensor.rank - 2
            if n_spatial > 0:
                size_dims = interpolate_size_arg(node, n_spatial, env, context, module_specs)
                scale = interpolate_scale_arg(node, n_spatial)
                if size_dims is not None or scale is not None:
                    result = infer_interpolate(tensor, size_dims, scale)
                    if result is not None:
                        return result
    callee_leaf = callee_name.split(".")[-1]
    if callee_leaf in REDUCTION_OPS and node.args:
        tensor = eval_expr(node.args[0], env, context, module_specs)
        if isinstance(tensor, TensorValue):
            rdim = reduction_dim(node, arg_offset=1)
            keepdim = reduction_keepdim(node, positional_index=2)
            return infer_reduction(tensor, rdim, keepdim)
    if callee_leaf in FUNCTIONAL_PASSTHROUGH and node.args:
        first_arg = eval_expr(node.args[0], env, context, module_specs)
        if isinstance(first_arg, TensorValue):
            return first_arg
    if callee_leaf == "one_hot" and node.args:
        tensor = eval_expr(node.args[0], env, context, module_specs)
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
    if callee_leaf == "topk" and node.args:
        tensor = eval_expr(node.args[0], env, context, module_specs)
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
    if callee_leaf == "bincount" and node.args:
        return TensorValue(TensorShape((UnknownDim("?"),)))
    if callee_leaf == "diagonal" and node.args:
        tensor = eval_expr(node.args[0], env, context, module_specs)
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
    if callee_leaf in LIKE_OPS and node.args:
        first_arg = eval_expr(node.args[0], env, context, module_specs)
        if isinstance(first_arg, TensorValue):
            return TensorValue(first_arg.shape)
    if callee_leaf in TENSOR_CONSTRUCTORS:
        dims = constructor_size(node, callee_leaf, env, context, module_specs)
        if dims is not None:
            return TensorValue(TensorShape(tuple(dims)))
    if callee_leaf == "arange" and node.args:
        arange_len = arange_length(node)
        if arange_len is not None:
            return TensorValue(TensorShape((ConstantDim(arange_len),)))
        arange_dim: Dim = (
            size_to_dim(node.args[0], env, context, module_specs)
            if len(node.args) == 1
            else UnknownDim("?")
        )
        return TensorValue(TensorShape((arange_dim,)))
    if callee_name.endswith("scaled_dot_product_attention") and node.args:
        q_val = eval_expr(node.args[0], env, context, module_specs)
        if isinstance(q_val, TensorValue):
            return q_val
    if callee_leaf == "split" and len(node.args) >= 2:
        tensor_arg = eval_expr(node.args[0], env, context, module_specs)
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
    module_alias_result = eval_named_module_alias_call(node, env, context, module_specs)
    if module_alias_result is not None:
        return module_alias_result
    signature_result = eval_signature_call(node, env, context, module_specs)
    if signature_result is not None:
        return signature_result
    maybe_warn_unannotated_function_call(node, env, context, module_specs)
    return None


def dim_from_expr(
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
        left = dim_from_expr(node.left, env, context, module_specs)
        right = dim_from_expr(node.right, env, context, module_specs)
        if left is not None and right is not None:
            return dim_binop(node.op, left, right)
    value = eval_expr(node, env, context, module_specs)
    if isinstance(value, IntegerValue):
        if value.value is not None:
            return value.value
        if isinstance(node, ast.Name):
            return SymbolicDim(node.id)
        return UnknownDim("?")
    if isinstance(value, (ConstantDim, ExpressionDim, SymbolicDim, UnknownDim)):
        return value
    if value is None and isinstance(node, ast.Name):
        return SymbolicDim(node.id)
    return None


def reshape_from_args(
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
        requested_dim = dim_from_expr(arg, env, context, module_specs)
        if requested_dim is None:
            context.error(node, "TSF1004", "Unsupported reshape dimension expression.")
            return None
        requested.append(requested_dim)
    return infer_reshape(tensor, tuple(requested))


def tensor_sequence(
    node: ast.AST,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> tuple[TensorValue, ...] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[TensorValue] = []
    for element in node.elts:
        value = eval_expr(element, env, context, module_specs)
        if not isinstance(value, TensorValue):
            return None
        values.append(value)
    return tuple(values)


def size_to_dim(
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
    result = eval_expr(node, env, context, module_specs)
    if isinstance(result, IntegerValue) and result.value is not None:
        return ConstantDim(result.value)
    if isinstance(result, (ConstantDim, SymbolicDim, ExpressionDim, UnknownDim)):
        return result
    return UnknownDim("?")


def constructor_size(
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
            return [size_to_dim(e, env, context, module_specs) for e in size_arg.elts]
        return [size_to_dim(size_arg, env, context, module_specs)]
    if not node.args:
        for kw in node.keywords:
            if kw.arg == "size" and isinstance(kw.value, (ast.Tuple, ast.List)):
                return [size_to_dim(e, env, context, module_specs) for e in kw.value.elts]
        return None
    if len(node.args) == 1 and isinstance(node.args[0], (ast.Tuple, ast.List)):
        return [size_to_dim(e, env, context, module_specs) for e in node.args[0].elts]
    return [size_to_dim(a, env, context, module_specs) for a in node.args]


def interpolate_size_arg(
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
    val = eval_expr(size_node, env, context, module_specs)
    if isinstance(val, ShapeTupleValue):
        dims = val.dims
        return tuple(dims[-n_spatial:]) if len(dims) >= n_spatial else None
    single = int_from_ast(size_node)
    if single is not None:
        return tuple(ConstantDim(single) for _ in range(n_spatial))
    if isinstance(size_node, (ast.Tuple, ast.List)):
        result_dims: list[Dim] = []
        for elt in size_node.elts:
            v = int_from_ast(elt)
            result_dims.append(ConstantDim(v) if v is not None else UnknownDim("?"))
        return tuple(result_dims)
    return tuple(UnknownDim("?") for _ in range(n_spatial))


def interpolate_scale_arg(node: ast.Call, n_spatial: int) -> tuple[float, ...] | None:
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

"""Tensor-method dispatcher: ``eval_tensor_method`` plus the ``expand`` helper.

Recursion: ``eval_tensor_method`` recurses on argument expressions via
``eval_expr`` (top-level import from ``expressions``) and shares the
``reshape_from_args`` / ``size_to_dim`` helpers with the call dispatcher
(top-level import from ``calls``). The reverse edge — ``calls`` →
``tensor_methods`` for ``eval_tensor_method`` — is a late import inside
``eval_call``, which breaks the cycle.
"""

from __future__ import annotations

import ast

from torchshapeflow.analysis_context import ModuleContext
from torchshapeflow.analyzer.calls import reshape_from_args, size_to_dim
from torchshapeflow.analyzer.constants import (
    NON_TENSOR_METHODS,
    PASSTHROUGH_METHODS,
    REDUCTION_OPS,
)
from torchshapeflow.analyzer.expressions import eval_expr
from torchshapeflow.analyzer.statements import emit_matmul_mismatch, emit_mm_mismatch
from torchshapeflow.ast_helpers import (
    infer_repeat_call,
    int_or_tuple,
    keyword_int,
    keyword_or_default,
    positional_int,
    reduction_dim,
    reduction_keepdim,
    split_from_call,
)
from torchshapeflow.model import (
    Dim,
    IntegerValue,
    ModuleSpec,
    TensorShape,
    TensorValue,
    UnknownDim,
    Value,
)
from torchshapeflow.rules import (
    infer_chunk,
    infer_diagonal,
    infer_flatten,
    infer_index_select,
    infer_matmul,
    infer_mm,
    infer_movedim,
    infer_permute,
    infer_reduction,
    infer_size,
    infer_squeeze,
    infer_topk,
    infer_transpose,
    infer_unsqueeze,
)
from torchshapeflow.rules.common import int_from_ast


def eval_tensor_method(
    tensor: TensorValue,
    node: ast.Call,
    context: ModuleContext,
    env: dict[str, Value],
    module_specs: dict[str, ModuleSpec],
) -> Value | None:
    assert isinstance(node.func, ast.Attribute)
    name = node.func.attr
    if name in {"reshape", "view"}:
        result = reshape_from_args(tensor, node.args, context, node, env, module_specs)
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
        right = eval_expr(node.args[0], env, context, module_specs)
        if isinstance(right, TensorValue):
            result = infer_matmul(tensor, right)
            if result is None:
                emit_matmul_mismatch(context, node, "matmul", tensor, right)
            return result
    if name == "mm" and node.args:
        right = eval_expr(node.args[0], env, context, module_specs)
        if isinstance(right, TensorValue):
            result = infer_mm(tensor, right)
            if result is None:
                emit_mm_mismatch(context, node, tensor, right)
            return result
    if name in REDUCTION_OPS:
        rdim = reduction_dim(node, arg_offset=0)
        keepdim = reduction_keepdim(node, positional_index=1)
        return infer_reduction(tensor, rdim, keepdim)
    if name in PASSTHROUGH_METHODS:
        return tensor
    if name == "expand" and node.args:
        return infer_expand(tensor, node, env, context, module_specs)
    if name == "expand_as" and node.args:
        other = eval_expr(node.args[0], env, context, module_specs)
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
            idx_val = eval_expr(node.args[1], env, context, module_specs)
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
    if name in NON_TENSOR_METHODS:
        return None
    if context.in_annotated_function:
        context.error(
            node,
            "TSF2001",
            f"Unsupported tensor method '.{name}' — shape not tracked.",
            severity="warning",
        )
    return None


def infer_expand(
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
            result_dims.append(size_to_dim(size_node, env, context, module_specs))
    return TensorValue(TensorShape(tuple(result_dims)))

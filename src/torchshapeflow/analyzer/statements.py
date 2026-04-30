"""Statement-level walker.

Dispatches each statement in a function body: alias bindings, assignments
(plain / annotated / augmented), returns, expression statements, and
``if``/``else`` flow-merging.
"""

from __future__ import annotations

import ast

from torchshapeflow.analysis_context import ModuleContext
from torchshapeflow.analyzer.functions import unpack_tensor_tuple
from torchshapeflow.ast_helpers import alias_target_node, shapes_definitely_mismatch
from torchshapeflow.index import extract_alias_binding
from torchshapeflow.model import (
    Conv2dSpec,
    CustomModuleSpec,
    Dim,
    EmbeddingSpec,
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
    TensorShape,
    TensorTupleValue,
    TensorValue,
    TupleValue,
    UnknownDim,
    Value,
    render_dim,
)
from torchshapeflow.parser import AnnotationParseError, parse_tensor_annotation
from torchshapeflow.rules import infer_binary_broadcast


def analyze_statement(
    statement: ast.stmt,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
    aliases: dict[str, TensorValue],
) -> None:
    from torchshapeflow.analyzer.expressions import eval_expr as _eval_expr

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
            target = alias_target_node(statement)
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
                unpack_tensor_tuple(list(assign_target.elts), value.tensors, env, context)
            elif isinstance(assign_target, ast.Tuple) and isinstance(value, TupleValue):
                unpack_tensor_tuple(list(assign_target.elts), value.items, env, context)
            elif isinstance(assign_target, ast.Tuple) and isinstance(value, ShapeTupleValue):
                for t_elt, dim in zip(assign_target.elts, value.dims, strict=False):
                    if isinstance(t_elt, ast.Name):
                        env[t_elt.id] = dim
            else:
                bind_target(assign_target, value, env, context)
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
        maybe_hover_alias_reference(statement.annotation, aliases, context)
        value = (
            _eval_expr(statement.value, env, context, module_specs)
            if statement.value is not None
            else None
        )
        if declared is not None:
            if isinstance(value, TensorValue) and shapes_definitely_mismatch(
                declared.shape, value.shape
            ):
                context.error(
                    statement,
                    "TSF1011",
                    f"Assigned shape {value.shape} does not match declared {declared.shape}.",
                )
            bind_target(statement.target, declared, env, context)
            return
        bind_target(statement.target, value, env, context)
        return
    if isinstance(statement, ast.Return) and statement.value is not None:
        actual = _eval_expr(statement.value, env, context, module_specs)
        context.collected_returns.append(actual if isinstance(actual, TensorValue) else None)
        if isinstance(actual, TensorValue):
            if not isinstance(statement.value, ast.Name):
                context.hover("<return>", statement.value, actual)
            if context.return_shape is not None and shapes_definitely_mismatch(
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
        analyze_if(statement, env, context, module_specs, aliases)
        return


def analyze_if(
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
        analyze_statement(stmt, env_then, context, module_specs, aliases_then)
    if node.orelse:
        env_else: dict[str, Value] = dict(pre_env)
        aliases_else: dict[str, TensorValue] = dict(pre_aliases)
        for stmt in node.orelse:
            analyze_statement(stmt, env_else, context, module_specs, aliases_else)
        merge_envs(env, pre_env, env_then, env_else)
        merge_aliases(aliases, pre_aliases, aliases_then, aliases_else)
    else:
        # No else: take the ``if`` body environment (pragmatically useful).
        env.update(env_then)
        aliases.clear()
        aliases.update(aliases_then)


def merge_aliases(
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


def maybe_hover_alias_reference(
    annotation: ast.AST,
    aliases: dict[str, TensorValue],
    context: ModuleContext,
) -> None:
    if isinstance(annotation, ast.Name):
        alias_value = aliases.get(annotation.id)
        if alias_value is not None:
            context.hover_alias(annotation.id, annotation, alias_value)


def merge_envs(
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


def bind_target(
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


def emit_matmul_mismatch(
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


def emit_mm_mismatch(
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

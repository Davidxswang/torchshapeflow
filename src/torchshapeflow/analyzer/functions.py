"""Function-level analysis: signature hover emission, return-annotation
suggestion, init-time tensor capture, and the per-function entry point used
by the module/class walkers.
"""

from __future__ import annotations

import ast
import copy

from torchshapeflow.analysis_context import ModuleContext
from torchshapeflow.analyzer.constants import PASSTHROUGH_METHODS
from torchshapeflow.ast_helpers import body_terminates_with_return, contains_top_level_yield
from torchshapeflow.model import (
    ConstantDim,
    ModuleSpec,
    SymbolicDim,
    TensorShape,
    TensorTupleValue,
    TensorValue,
    TupleValue,
    Value,
)
from torchshapeflow.parser import AnnotationParseError, parse_tensor_annotation
from torchshapeflow.report import HoverFact, Suggestion


def analyze_function(
    function: ast.FunctionDef,
    context: ModuleContext,
    module_specs: dict[str, ModuleSpec],
) -> None:
    # Late import: statements.py imports back into expressions/__init__.py
    # via _eval_expr; resolving at call time avoids the cycle.
    from torchshapeflow.analyzer.statements import analyze_statement as _analyze_statement

    # Snapshot error-severity diagnostic count *before* annotation parsing so
    # TSF1001 errors on malformed param annotations are counted against this
    # function alongside body errors. If the snapshot happened after
    # collect_function_annotations, a bad annotation on one param would slip
    # past the suggest guard when a sibling param's return path was clean.
    errors_before = sum(1 for d in context.diagnostics if d.severity == "error")
    env, local_aliases, tensor_params, return_shape = collect_function_annotations(
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
    for statement in function.body:
        _analyze_statement(statement, env, context, module_specs, local_aliases)
    # Emit a signature hover on the function name if any tensor params are present.
    if tensor_params:
        emit_signature_hover(
            function, tensor_params, context.return_shape, context.collected_returns, context
        )
    maybe_suggest_return_annotation(
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


def emit_function_annotation_hovers(function: ast.FunctionDef, context: ModuleContext) -> None:
    _, _, tensor_params, return_shape = collect_function_annotations(function, context)
    if tensor_params:
        emit_signature_hover(function, tensor_params, return_shape, [], context)


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


def maybe_suggest_return_annotation(
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
    - ``body_terminates_with_return`` proves every exit path returns a value
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
    if contains_top_level_yield(function.body):
        return
    if not body_terminates_with_return(function.body):
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


def collect_function_annotations(
    function: ast.FunctionDef,
    context: ModuleContext,
    *,
    emit_hovers: bool = True,
) -> tuple[
    dict[str, Value], dict[str, TensorValue], list[tuple[str, TensorValue]], TensorValue | None
]:
    from torchshapeflow.analyzer.statements import (
        maybe_hover_alias_reference as _maybe_hover_alias_reference,
    )

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


def init_tensor_from_expr(node: ast.AST, env: dict[str, Value]) -> TensorValue | None:
    value = env.get(node.id) if isinstance(node, ast.Name) else None
    if isinstance(value, TensorValue):
        return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = init_tensor_from_expr(node.func.value, env)
        if base is not None and node.func.attr in PASSTHROUGH_METHODS:
            return base
    return None


def is_register_buffer_call(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == "register_buffer"
    )


def register_buffer_binding(
    statement: ast.stmt,
    env: dict[str, Value],
) -> tuple[str, TensorValue] | None:
    if not is_register_buffer_call(statement):
        return None
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    if len(call.args) < 2:
        return None
    name_node = call.args[0]
    if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
        return None
    tensor_value = init_tensor_from_expr(call.args[1], env)
    if tensor_value is None:
        return None
    return name_node.value, tensor_value


def emit_signature_hover(
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


def unpack_tensor_tuple(
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
    from torchshapeflow.analyzer.statements import bind_target as _bind_target

    value_idx = 0
    for elt in elts:
        if value_idx >= len(values):
            break
        current = values[value_idx]
        if isinstance(elt, ast.Tuple):
            if isinstance(current, TensorTupleValue):
                unpack_tensor_tuple(list(elt.elts), current.tensors, env, context)
            elif isinstance(current, TupleValue):
                unpack_tensor_tuple(list(elt.elts), current.items, env, context)
            else:
                _bind_target(elt, current, env, context)
        else:
            _bind_target(elt, current, env, context)
        value_idx += 1

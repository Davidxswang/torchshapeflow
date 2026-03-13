from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from torchshapeflow.diagnostics import Diagnostic, Severity
from torchshapeflow.model import (
    ConstantDim,
    Conv2dSpec,
    Dim,
    ExpressionDim,
    IntegerValue,
    LinearSpec,
    ShapeTupleValue,
    SymbolicDim,
    TensorValue,
    UnknownDim,
    Value,
    make_dim,
)
from torchshapeflow.parser import AnnotationParseError, parse_source, parse_tensor_annotation
from torchshapeflow.report import FileReport, HoverFact
from torchshapeflow.rules import (
    infer_binary_broadcast,
    infer_cat,
    infer_conv2d,
    infer_flatten,
    infer_linear,
    infer_matmul,
    infer_permute,
    infer_reshape,
    infer_size,
    infer_squeeze,
    infer_stack,
    infer_subscript,
    infer_transpose,
    infer_unsqueeze,
)
from torchshapeflow.rules.common import int_from_ast, qualified_name


@dataclass
class ModuleContext:
    path: Path
    diagnostics: list[Diagnostic] = field(default_factory=list)
    hovers: list[HoverFact] = field(default_factory=list)

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


def analyze_path(path: Path) -> FileReport:
    source = path.read_text(encoding="utf-8")
    return analyze_source(source, path)


def analyze_source(source: str, path: Path) -> FileReport:
    module = parse_source(source, str(path))
    context = ModuleContext(path=path)
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


def _collect_class_specs(module: ast.Module) -> dict[str, dict[str, LinearSpec | Conv2dSpec]]:
    specs: dict[str, dict[str, LinearSpec | Conv2dSpec]] = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        class_specs: dict[str, LinearSpec | Conv2dSpec] = {}
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                for statement in child.body:
                    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                        target = statement.targets[0]
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            spec = _parse_module_spec(statement.value)
                            if spec is not None:
                                class_specs[target.attr] = spec
        if class_specs:
            specs[node.name] = class_specs
    return specs


def _parse_module_spec(node: ast.AST) -> LinearSpec | Conv2dSpec | None:
    if not isinstance(node, ast.Call):
        return None
    name = qualified_name(node.func)
    if name.endswith("Linear") and len(node.args) >= 2:
        in_features = int_from_ast(node.args[0])
        out_features = int_from_ast(node.args[1])
        if in_features is not None and out_features is not None:
            return LinearSpec(in_features=in_features, out_features=out_features)
    if name.endswith("Conv2d") and len(node.args) >= 3:
        in_channels = int_from_ast(node.args[0])
        out_channels = int_from_ast(node.args[1])
        kernel_size = _int_pair(node.args[2])
        stride = _int_pair(_keyword_or_default(node, "stride"), default=(1, 1))
        padding = _int_pair(_keyword_or_default(node, "padding"), default=(0, 0))
        dilation = _int_pair(_keyword_or_default(node, "dilation"), default=(1, 1))
        if None not in (in_channels, out_channels, kernel_size, stride, padding, dilation):
            assert in_channels is not None
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


def _analyze_function(
    function: ast.FunctionDef,
    context: ModuleContext,
    module_specs: dict[str, LinearSpec | Conv2dSpec],
) -> None:
    env: dict[str, Value] = {}
    for argument in function.args.args:
        if argument.arg == "self":
            continue
        if argument.annotation is None:
            continue
        try:
            tensor = parse_tensor_annotation(argument.annotation)
        except AnnotationParseError as error:
            context.error(argument, "TSF1001", error.message)
            continue
        if tensor is not None:
            env[argument.arg] = tensor
            context.hover(argument.arg, argument, tensor)
    for statement in function.body:
        _analyze_statement(statement, env, context, module_specs)


def _analyze_statement(
    statement: ast.stmt,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, LinearSpec | Conv2dSpec],
) -> None:
    if isinstance(statement, ast.Assign):
        value = _eval_expr(statement.value, env, context, module_specs)
        for target in statement.targets:
            _bind_target(target, value, env, context)
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
        _eval_expr(statement.value, env, context, module_specs)
        return
    if isinstance(statement, ast.Expr):
        _eval_expr(statement.value, env, context, module_specs)


def _bind_target(
    target: ast.AST,
    value: Value | Dim | None,
    env: dict[str, Value],
    context: ModuleContext,
) -> None:
    if not isinstance(target, ast.Name):
        return
    if isinstance(value, (TensorValue, ShapeTupleValue, IntegerValue, LinearSpec, Conv2dSpec)):
        env[target.id] = value
        if isinstance(value, TensorValue):
            context.hover(target.id, target, value)


def _eval_expr(
    node: ast.AST,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, LinearSpec | Conv2dSpec],
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
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            # Only direct `self.attr` access is tracked; aliases such as
            # `m = self; m.attr(...)` are not supported.
            return module_specs.get(node.attr)
        return None
    if isinstance(node, ast.Subscript):
        base = _eval_expr(node.value, env, context, module_specs)
        if isinstance(base, (TensorValue, ShapeTupleValue)):
            return infer_subscript(base, node)
        return None
    _element_wise_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    if isinstance(node, ast.BinOp) and isinstance(node.op, _element_wise_ops):
        left = _eval_expr(node.left, env, context, module_specs)
        right = _eval_expr(node.right, env, context, module_specs)
        if isinstance(left, TensorValue) and isinstance(right, TensorValue):
            result = infer_binary_broadcast(left, right)
            if result is None:
                context.error(node, "TSF1006", "Broadcasting incompatibility.")
            return result
        return None
    if isinstance(node, ast.Call):
        return _eval_call(node, env, context, module_specs)
    if isinstance(node, ast.Tuple):
        return None
    return None


def _eval_call(
    node: ast.Call,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, LinearSpec | Conv2dSpec],
) -> Value | None:
    callee_name = qualified_name(node.func)
    if isinstance(node.func, ast.Attribute):
        owner: Value | Dim | None
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            owner = module_specs.get(node.func.attr)
        else:
            owner = _eval_expr(node.func.value, env, context, module_specs)
        if isinstance(owner, TensorValue):
            return _eval_tensor_method(owner, node, context, env, module_specs)
        if isinstance(owner, LinearSpec):
            argument = _eval_expr(node.args[0], env, context, module_specs) if node.args else None
            if isinstance(argument, TensorValue):
                result = infer_linear(owner, argument)
                if result is None:
                    context.error(node, "TSF1007", "nn.Linear input shape mismatch.")
                return result
        if isinstance(owner, Conv2dSpec):
            argument = _eval_expr(node.args[0], env, context, module_specs) if node.args else None
            if isinstance(argument, TensorValue):
                result = infer_conv2d(owner, argument)
                if result is None:
                    context.error(node, "TSF1007", "nn.Conv2d input shape mismatch.")
                return result
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
    return None


def _eval_tensor_method(
    tensor: TensorValue,
    node: ast.Call,
    context: ModuleContext,
    env: dict[str, Value],
    module_specs: dict[str, LinearSpec | Conv2dSpec],
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
    return None


def _reshape_from_args(
    tensor: TensorValue,
    args: Sequence[ast.expr],
    context: ModuleContext,
    node: ast.AST,
    env: dict[str, Value],
    module_specs: dict[str, LinearSpec | Conv2dSpec],
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
    module_specs: dict[str, LinearSpec | Conv2dSpec],
) -> Dim | int | None:
    integer = int_from_ast(node)
    if integer is not None:
        return integer
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return make_dim(node.value)
    value = _eval_expr(node, env, context, module_specs)
    if isinstance(value, IntegerValue) and value.value is not None:
        return value.value
    if isinstance(value, (ConstantDim, ExpressionDim, SymbolicDim, UnknownDim)):
        return value
    return None


def _tensor_sequence(
    node: ast.AST,
    env: dict[str, Value],
    context: ModuleContext,
    module_specs: dict[str, LinearSpec | Conv2dSpec],
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

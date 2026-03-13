from __future__ import annotations

import ast
from dataclasses import dataclass

from torchshapeflow.model import Dim, TensorShape, TensorValue, make_dim


@dataclass(frozen=True)
class AnnotationParseError(Exception):
    message: str


_ANNOTATED_NAMES: frozenset[str] = frozenset(
    {"Annotated", "typing.Annotated", "typing_extensions.Annotated"}
)


def parse_source(source: str, path: str = "<memory>") -> ast.Module:
    """Parse Python source code into an AST module.

    Args:
        source: Raw Python source text.
        path: Filename reported in syntax errors. Defaults to "<memory>".

    Returns:
        Parsed ast.Module.
    """
    return ast.parse(source, filename=path)


def parse_tensor_annotation(node: ast.AST) -> TensorValue | None:
    """Extract a TensorValue from an Annotated[torch.Tensor, Shape(...)] annotation node.

    Args:
        node: An AST node that may represent an annotated tensor type.

    Returns:
        TensorValue carrying the declared shape, or None if the node is not an
        Annotated tensor annotation (e.g. it annotates a non-Tensor type).

    Raises:
        AnnotationParseError: If the node looks like an Annotated tensor annotation but is
            malformed (missing Shape metadata, or invalid Shape arguments).
    """
    if not isinstance(node, ast.Subscript):
        return None
    if not _is_annotated(node.value):
        return None
    parts = _subscript_elements(node)
    if len(parts) < 2:
        raise AnnotationParseError("Annotated must include a base type and Shape metadata.")
    base, *metadata = parts
    if not _is_tensor_type(base):
        return None
    for item in metadata:
        if _qualified_name(item) == "Shape":
            dims = tuple(_parse_shape_arg(arg) for arg in _call_args(item))
            return TensorValue(TensorShape(dims))
    raise AnnotationParseError("Annotated tensor is missing Shape(...) metadata.")


def _parse_shape_arg(node: ast.AST) -> Dim:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, str)):
        return make_dim(node.value)
    raise AnnotationParseError("Shape metadata only supports string or integer dimensions.")


def _is_annotated(node: ast.AST) -> bool:
    return _qualified_name(node) in _ANNOTATED_NAMES


def _is_tensor_type(node: ast.AST) -> bool:
    qualified = _qualified_name(node)
    return qualified == "torch.Tensor" or qualified.endswith(".Tensor") or qualified == "Tensor"


def _subscript_elements(node: ast.Subscript) -> list[ast.AST]:
    slice_node = node.slice
    if isinstance(slice_node, ast.Tuple):
        return list(slice_node.elts)
    return [slice_node]


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value)
        if not base:
            return node.attr
        return f"{base}.{node.attr}"
    if isinstance(node, ast.Call):
        return _qualified_name(node.func)
    return ""


def _call_args(node: ast.AST) -> list[ast.AST]:
    if not isinstance(node, ast.Call):
        raise AnnotationParseError("Expected Shape(...) metadata.")
    return list(node.args)

from __future__ import annotations

import ast

import pytest

from torchshapeflow.parser import (
    AnnotationParseError,
    _is_annotated,
    _is_tensor_type,
    parse_source,
    parse_tensor_annotation,
)


def _expr(src: str) -> ast.AST:
    """Parse a single expression into an AST node."""
    return ast.parse(src, mode="eval").body


# ---------------------------------------------------------------------------
# parse_source
# ---------------------------------------------------------------------------


def test_parse_source_returns_module() -> None:
    module = parse_source("x = 1")
    assert isinstance(module, ast.Module)


def test_parse_source_custom_path() -> None:
    module = parse_source("x = 1", path="test.py")
    assert isinstance(module, ast.Module)


# ---------------------------------------------------------------------------
# parse_tensor_annotation
# ---------------------------------------------------------------------------


def test_parse_annotation_string_dims() -> None:
    node = _expr("Annotated[torch.Tensor, Shape('B', 'C')]")
    result = parse_tensor_annotation(node)
    assert result is not None
    assert str(result.shape) == "[B, C]"


def test_parse_annotation_mixed_dims() -> None:
    node = _expr("Annotated[torch.Tensor, Shape('B', 3, 32, 32)]")
    result = parse_tensor_annotation(node)
    assert result is not None
    assert str(result.shape) == "[B, 3, 32, 32]"


def test_parse_annotation_typing_qualified() -> None:
    node = _expr("typing.Annotated[torch.Tensor, Shape('B',)]")
    result = parse_tensor_annotation(node)
    assert result is not None
    assert str(result.shape) == "[B]"


def test_parse_annotation_string_shorthand() -> None:
    node = _expr("Annotated[torch.Tensor, 'B C H W']")
    result = parse_tensor_annotation(node)
    assert result is not None
    assert str(result.shape) == "[B, C, H, W]"


def test_parse_annotation_string_shorthand_mixed_dims() -> None:
    node = _expr("Annotated[torch.Tensor, 'B 3 224 224']")
    result = parse_tensor_annotation(node)
    assert result is not None
    assert str(result.shape) == "[B, 3, 224, 224]"


def test_parse_annotation_not_subscript_returns_none() -> None:
    node = _expr("torch.Tensor")
    assert parse_tensor_annotation(node) is None


def test_parse_annotation_not_annotated_returns_none() -> None:
    node = _expr("List[torch.Tensor]")
    assert parse_tensor_annotation(node) is None


def test_parse_annotation_non_tensor_base_returns_none() -> None:
    node = _expr("Annotated[int, Shape('B',)]")
    assert parse_tensor_annotation(node) is None


def test_parse_annotation_missing_shape_raises() -> None:
    node = _expr("Annotated[torch.Tensor, SomeOtherMeta()]")
    with pytest.raises(AnnotationParseError):
        parse_tensor_annotation(node)


def test_parse_annotation_too_few_args_raises() -> None:
    # Annotated with only one type argument (no metadata)
    node = _expr("Annotated[torch.Tensor]")
    # This is syntactically valid Python but semantically wrong
    with pytest.raises(AnnotationParseError):
        parse_tensor_annotation(node)


# ---------------------------------------------------------------------------
# _is_annotated
# ---------------------------------------------------------------------------


def test_is_annotated_bare_name() -> None:
    assert _is_annotated(_expr("Annotated"))


def test_is_annotated_typing_qualified() -> None:
    assert _is_annotated(_expr("typing.Annotated"))


def test_is_annotated_typing_extensions_qualified() -> None:
    assert _is_annotated(_expr("typing_extensions.Annotated"))


def test_is_annotated_rejects_similar_name() -> None:
    # Should NOT match arbitrary names that merely end with "Annotated"
    assert not _is_annotated(_expr("MyAnnotated"))
    assert not _is_annotated(_expr("NotAnnotated"))


def test_is_annotated_rejects_list() -> None:
    assert not _is_annotated(_expr("List"))


# ---------------------------------------------------------------------------
# _is_tensor_type
# ---------------------------------------------------------------------------


def test_is_tensor_type_qualified() -> None:
    assert _is_tensor_type(_expr("torch.Tensor"))


def test_is_tensor_type_bare() -> None:
    # Common usage: from torch import Tensor
    assert _is_tensor_type(_expr("Tensor"))


def test_is_tensor_type_other_module_qualified() -> None:
    # e.g. numpy.Tensor — matches *.Tensor convention
    assert _is_tensor_type(_expr("mylib.Tensor"))


def test_is_tensor_type_rejects_non_tensor() -> None:
    assert not _is_tensor_type(_expr("int"))
    assert not _is_tensor_type(_expr("torch.nn.Module"))

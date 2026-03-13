from __future__ import annotations

import ast

from torchshapeflow.model import ConstantDim, SymbolicDim
from torchshapeflow.parser import parse_tensor_annotation


def test_parse_shape_annotation() -> None:
    node = ast.parse(
        'x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]',
        mode="single",
    ).body[0]
    assert isinstance(node, ast.AnnAssign)
    tensor = parse_tensor_annotation(node.annotation)
    assert tensor is not None
    assert tensor.shape.dims == (
        SymbolicDim("B"),
        ConstantDim(3),
        ConstantDim(32),
        ConstantDim(32),
    )

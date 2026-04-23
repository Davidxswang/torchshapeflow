from __future__ import annotations

from pathlib import Path

from torchshapeflow.analyzer import analyze_source


def test_suggest_return_annotation_symbolic_shape() -> None:
    """Params annotated, no return annotation, body tracks — suggestion emitted."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def scores(
    q: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
    k: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
):
    return q @ k.transpose(-2, -1)
"""
    report = analyze_source(source, Path("m.py"))
    assert len(report.suggestions) == 1
    sug = report.suggestions[0]
    assert sug.kind == "return_annotation"
    assert sug.function == "scores"
    assert sug.shape == "[B, H, T, T]"
    assert sug.annotation == 'Annotated[torch.Tensor, Shape("B", "H", "T", "T")]'


def test_suggest_return_annotation_mixed_const_symbolic() -> None:
    """Integer dims render without quotes; symbolic dims stay quoted."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(768, 256)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 768)]):
        return self.fc(x)
"""
    report = analyze_source(source, Path("m.py"))
    # forward should get a return suggestion; __init__ is not annotated and is skipped.
    suggestions = [s for s in report.suggestions if s.function == "forward"]
    assert len(suggestions) == 1
    assert suggestions[0].annotation == 'Annotated[torch.Tensor, Shape("B", "T", 256)]'


def test_no_suggestion_when_return_annotation_present() -> None:
    """If the user already declared a return annotation, don't propose one."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(
    x: Annotated[torch.Tensor, Shape("B", "T")],
) -> Annotated[torch.Tensor, Shape("B", "T")]:
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_no_suggestion_when_params_unannotated() -> None:
    """Without any parameter annotations the user hasn't opted in; stay silent."""
    source = """
import torch


def fn(x):
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_no_suggestion_when_return_shapes_diverge() -> None:
    """Two return statements with different shapes — ambiguous, skip."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(
    a: Annotated[torch.Tensor, Shape("B", 3)],
    b: Annotated[torch.Tensor, Shape("B", 4)],
    pick_a: bool,
):
    if pick_a:
        return a
    return b
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_no_suggestion_when_return_shape_contains_expression_dim() -> None:
    """ExpressionDim can't round-trip through Shape(...), so skip."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(
    x: Annotated[torch.Tensor, Shape("B", "T", "D")],
):
    # reshape produces an ExpressionDim "B*T" in the last dim combine — not expressible.
    return x.reshape(x.shape[0] * x.shape[1], x.shape[2])
"""
    report = analyze_source(source, Path("m.py"))
    # The inferred return shape is [B*T, D]; B*T is an ExpressionDim — skip.
    assert report.suggestions == []


def test_no_suggestion_when_return_is_non_tensor() -> None:
    """A function returning a tuple / int / None is not a single-tensor return."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B", "T")]):
    return x, x
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_suggestion_position_points_at_function_name() -> None:
    """Suggestion position uses the function-name convention (like signature hovers)."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def my_func(x: Annotated[torch.Tensor, Shape("B",)]):
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert len(report.suggestions) == 1
    sug = report.suggestions[0]
    # `def ` is 4 chars, so the name token starts at column 5 (1-based).
    assert sug.line == 7
    assert sug.column == 5
    assert sug.end_column == 5 + len("my_func")


def test_suggestion_serializes_to_dict() -> None:
    """Suggestion.to_dict and FileReport.to_dict include the suggestion fields."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)]):
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert len(report.suggestions) == 1
    sug_payload = report.suggestions[0].to_dict()
    assert sug_payload["kind"] == "return_annotation"
    assert sug_payload["function"] == "fn"
    assert sug_payload["annotation"] == 'Annotated[torch.Tensor, Shape("B")]'
    report_payload = report.to_dict()
    assert "suggestions" in report_payload
    assert report_payload["suggestions"] == [sug_payload]

from __future__ import annotations

from pathlib import Path

from torchshapeflow.analyzer import analyze_source


def test_analyze_transformer_shapes() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def block(x: Annotated[torch.Tensor, Shape("B", "T", 768)]):
    q = x.reshape(x.shape[0], x.shape[1], 12, 64)
    k = q.permute(0, 2, 1, 3)
    return k
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "q" and hover.shape == "[B, T, 12, 64]" for hover in report.hovers)
    assert any(hover.name == "k" and hover.shape == "[B, 12, T, 64]" for hover in report.hovers)
    assert any(
        hover.name == "q"
        and hover.line == 7
        and hover.column == 5
        and hover.end_line == 7
        and hover.end_column == 6
        for hover in report.hovers
    )


def test_analyze_linear_and_conv() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.linear = nn.Linear(8 * 32 * 32, 10)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]):
        y = self.conv(x)
        z = y.flatten(1)
        return self.linear(z)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, 8, 32, 32]" for hover in report.hovers)
    assert any(hover.name == "z" and hover.shape == "[B, 8192]" for hover in report.hovers)


def test_bad_reshape_reports_error() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def bad(x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]):
    y = x.reshape(-1, -1)
    return y
"""
    report = analyze_source(source, Path("memory.py"))
    assert any(diagnostic.code == "TSF1004" for diagnostic in report.diagnostics)


def test_transformer_attention_scores() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def attention_scores(
    q: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
    k: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
):
    scores = q.matmul(k.transpose(-2, -1))
    return scores
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "scores" and hover.shape == "[B, H, T, T]" for hover in report.hovers)

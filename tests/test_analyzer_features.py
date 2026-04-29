from __future__ import annotations

from pathlib import Path

from torchshapeflow.analyzer import analyze_source

# ---------------------------------------------------------------------------
# Feature: if/else support
# ---------------------------------------------------------------------------


def test_if_else_same_shape_preserved() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def f(x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]):
    if True:
        y = x.flatten(1)
    else:
        y = x.flatten(1)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "y"), None)
    assert hover is not None
    assert hover.shape == "[B, 3072]"


def test_if_else_different_shape_unknown_dims() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def f(x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]):
    if True:
        y = x.reshape(x.shape[0], 3, 32, 32)
    else:
        y = x.reshape(x.shape[0], 3, 16, 64)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    # Multiple hovers for "y" exist (one per branch + merged). Take the last one (post-merge).
    y_hovers = [h for h in report.hovers if h.name == "y"]
    assert len(y_hovers) >= 1
    hover = y_hovers[-1]
    # B and 3 match, but 32 vs 16 and 32 vs 64 differ → [B, 3, ?, ?]
    assert hover.shape == "[B, 3, ?, ?]"


def test_if_no_else_variable_available() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def f(x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]):
    if True:
        y = x.flatten(1)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "y"), None)
    assert hover is not None
    assert hover.shape == "[B, 3072]"


def test_if_else_preexisting_variable_preserved() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def f(x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]):
    z = x.flatten(1)
    if True:
        pass
    else:
        pass
    return z
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "z"), None)
    assert hover is not None
    assert hover.shape == "[B, 3072]"


# ---------------------------------------------------------------------------
# Feature: non-literal constructor args from __init__ params
# ---------------------------------------------------------------------------


def test_init_param_variable_out_features_becomes_symbolic() -> None:
    """Variable out_features captured as SymbolicDim regardless of default value."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 32):
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 64)]):
        return self.linear(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "<return>"), None)
    assert hover is not None
    # out_dim is a variable name — captured as SymbolicDim("out_dim"), not resolved to 32.
    assert hover.shape == "[B, out_dim]"


def test_init_param_no_default_still_infers_symbolic() -> None:
    """Variable out_dim with no default still produces a symbolic output dim."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self, out_dim: int):
        self.linear = nn.Linear(64, out_dim)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 64)]):
        return self.linear(x)
"""
    report = analyze_source(source, Path("f.py"))
    # out_dim has no default but is a valid variable name — spec is created with SymbolicDim.
    hover = next((h for h in report.hovers if h.name == "<return>"), None)
    assert hover is not None
    assert hover.shape == "[B, out_dim]"


def test_init_param_mixed_literal_and_param() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self, hidden: int = 128):
        self.linear = nn.Linear(64, hidden)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 64)]):
        return self.linear(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "<return>"), None)
    assert hover is not None
    # hidden is a variable name — captured as SymbolicDim("hidden"), not resolved to 128.
    assert hover.shape == "[B, hidden]"


# ---------------------------------------------------------------------------
# Feature: symbolic unification across call sites
# ---------------------------------------------------------------------------


def test_symbolic_unification_consistent() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def helper(
    x: Annotated[torch.Tensor, Shape("B", "D")],
    y: Annotated[torch.Tensor, Shape("B", "E")],
) -> Annotated[torch.Tensor, Shape("B", "D")]:
    return x

def main(
    a: Annotated[torch.Tensor, Shape(8, 64)],
    b: Annotated[torch.Tensor, Shape(8, 32)],
):
    out = helper(a, b)
    return out
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "out"), None)
    assert hover is not None
    # B→8, D→64 substituted into return shape
    assert hover.shape == "[8, 64]"


def test_symbolic_unification_conflict() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def helper(
    x: Annotated[torch.Tensor, Shape("B", "D")],
    y: Annotated[torch.Tensor, Shape("B", "E")],
) -> Annotated[torch.Tensor, Shape("B", "D")]:
    return x

def main(
    a: Annotated[torch.Tensor, Shape(8, 64)],
    b: Annotated[torch.Tensor, Shape(16, 32)],
):
    out = helper(a, b)
    return out
"""
    report = analyze_source(source, Path("f.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1010"]
    assert len(errors) == 1
    assert "B" in errors[0].message
    # Conflicting dim becomes unknown in the return shape
    hover = next((h for h in report.hovers if h.name == "out"), None)
    assert hover is not None
    assert hover.shape == "[?, 64]"


# ---------------------------------------------------------------------------
# shape tuple unpacking:  a, b, c, d = x.shape
# ---------------------------------------------------------------------------


def test_shape_unpack_constant_dims() -> None:
    """Unpacking a shape with all constant dims seeds known integer values."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape(2, 4, 8)]):
    a, b, c = x.shape
    y = x.reshape(a, b * c)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[2, 32]" for hover in report.hovers)


def test_shape_unpack_symbolic_dims() -> None:
    """Unpacking a shape with symbolic dims binds variable names as symbolic dims."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", "N", "F")]):
    batch_size, lookback, num_counties, num_features = x.shape
    x_flat = x.reshape(batch_size, lookback, num_counties * num_features)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "x_flat" and hover.shape == "[B, T, N*F]" for hover in report.hovers)


def test_shape_unpack_partial() -> None:
    """Fewer unpack targets than shape dims — only the named variables are bound."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", 16, 32)]):
    batch, channels = x.shape
    y = x.reshape(batch, channels * 32)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    # channels=16, 32 is literal — product is 512; batch is the original dim B
    assert any(hover.name == "y" and hover.shape == "[B, 512]" for hover in report.hovers)

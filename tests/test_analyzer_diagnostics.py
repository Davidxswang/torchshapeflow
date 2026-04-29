from __future__ import annotations

from pathlib import Path

from torchshapeflow.analyzer import analyze_source

# ---------------------------------------------------------------------------
# movedim / mm
# ---------------------------------------------------------------------------


def test_movedim_tensor_method() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C", "H", "W")]):
    y = x.movedim(1, -1)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    # axis 1 (C) moved to last position → (B, H, W, C)
    assert any(hover.name == "y" and hover.shape == "[B, H, W, C]" for hover in report.hovers)


def test_torch_mm() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape(4, 8)],
       y: Annotated[torch.Tensor, Shape(8, 16)]):
    z = torch.mm(x, y)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "z" and hover.shape == "[4, 16]" for hover in report.hovers)


def test_torch_mm_incompatible_tsf1003() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape(4, 8)],
       y: Annotated[torch.Tensor, Shape(9, 16)]):
    z = torch.mm(x, y)
"""
    report = analyze_source(source, Path("memory.py"))
    assert any(d.code == "TSF1003" for d in report.diagnostics)


def test_tsf1003_mm_carries_structured_fields() -> None:
    """TSF1003 for torch.mm exposes expected/actual/hint for agents and editors."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape(4, 8)],
       y: Annotated[torch.Tensor, Shape(9, 16)]):
    z = torch.mm(x, y)
"""
    report = analyze_source(source, Path("memory.py"))
    mm_errors = [d for d in report.diagnostics if d.code == "TSF1003"]
    assert mm_errors, "expected at least one TSF1003 diagnostic"
    diag = mm_errors[0]
    assert diag.expected is not None and "rank-2" in diag.expected
    assert diag.actual is not None and "[4, 8]" in diag.actual and "[9, 16]" in diag.actual
    assert diag.hint is not None
    # Structured fields must round-trip through JSON dict output.
    payload = diag.to_dict()
    assert payload["expected"] == diag.expected
    assert payload["actual"] == diag.actual
    assert payload["hint"] == diag.hint


def test_tsf1007_linear_carries_structured_fields() -> None:
    """TSF1007 for nn.Linear exposes the expected/actual shapes and a hint."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(768, 256)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 512)]):
        return self.fc(x)
"""
    report = analyze_source(source, Path("memory.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1007"]
    assert errors, "expected a TSF1007 Linear mismatch"
    diag = errors[0]
    assert diag.expected == "last dim = 768"
    assert diag.actual is not None and "512" in diag.actual
    assert diag.hint is not None and "768" in diag.hint
    # The rendered prose message incorporates all structured fields.
    assert "expected last dim = 768" in diag.message
    assert "got" in diag.message
    assert "hint:" in diag.message


def test_tsf1007_conv2d_wrong_rank_structured_fields() -> None:
    """Conv2d against a non-rank-4 tensor reports rank and shape explicitly."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 3, "H")]):
        return self.conv(x)
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1007"]
    assert errors
    diag = errors[0]
    assert diag.expected == "rank-4 tensor (N, C, H, W)"
    assert diag.actual is not None and "rank-3" in diag.actual
    assert diag.hint is not None and "4D" in diag.hint


def test_tsf1007_conv2d_wrong_channels_structured_fields() -> None:
    """Conv2d with rank-4 input but mismatched channels dim."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 5, "H", "W")]):
        return self.conv(x)
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1007"]
    assert errors
    diag = errors[0]
    assert diag.expected == "channels dim = 3"
    assert diag.actual is not None and "channels dim = 5" in diag.actual
    assert diag.hint is not None and "3" in diag.hint


def test_tsf1007_pool2d_wrong_rank_structured_fields() -> None:
    """nn.MaxPool2d against a non-rank-4 tensor."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "C", "H")]):
        return self.pool(x)
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1007"]
    assert errors
    diag = errors[0]
    assert diag.expected == "rank-4 tensor (N, C, H, W)"
    assert diag.actual is not None and "rank-3" in diag.actual


def test_tsf1007_lstm_wrong_rank_structured_fields() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(64, 128)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", 64)]):
        return self.lstm(x)
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1007"]
    assert errors
    diag = errors[0]
    assert diag.expected is not None and "rank-3" in diag.expected
    assert diag.actual is not None and "rank-2" in diag.actual


def test_tsf1007_lstm_wrong_input_size_structured_fields() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(64, 128)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", "N", 32)]):
        return self.lstm(x)
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1007"]
    assert errors
    diag = errors[0]
    assert diag.expected == "last dim = 64"
    assert diag.actual is not None and "32" in diag.actual
    assert diag.hint is not None and "64" in diag.hint


def test_tsf1003_matmul_binop_structured_fields() -> None:
    """The `q @ k` binop form produces structured fields identical to torch.matmul."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(q: Annotated[torch.Tensor, Shape("B", "T", 64)],
       k: Annotated[torch.Tensor, Shape("B", 128, "T")]):
    return q @ k
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1003"]
    assert errors
    diag = errors[0]
    assert diag.expected is not None and "last dim of left" in diag.expected
    assert diag.actual is not None
    assert "[B, T, 64]" in diag.actual
    assert "[B, 128, T]" in diag.actual
    assert diag.hint is not None and "transpose" in diag.hint


def test_tsf1003_torch_matmul_call_structured_fields() -> None:
    """torch.matmul(a, b) call form."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(a: Annotated[torch.Tensor, Shape(4, 8)],
       b: Annotated[torch.Tensor, Shape(9, 16)]):
    return torch.matmul(a, b)
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1003"]
    assert errors
    diag = errors[0]
    assert diag.expected is not None
    assert diag.actual is not None and "[4, 8]" in diag.actual and "[9, 16]" in diag.actual
    assert diag.hint is not None


def test_tsf1003_tensor_method_matmul_structured_fields() -> None:
    """Tensor.matmul(other) method form."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(a: Annotated[torch.Tensor, Shape(4, 8)],
       b: Annotated[torch.Tensor, Shape(9, 16)]):
    return a.matmul(b)
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1003"]
    assert errors
    diag = errors[0]
    assert diag.actual is not None and "[4, 8]" in diag.actual and "[9, 16]" in diag.actual


def test_tsf1003_tensor_method_mm_structured_fields() -> None:
    """Tensor.mm(other) method form."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(a: Annotated[torch.Tensor, Shape(4, 8)],
       b: Annotated[torch.Tensor, Shape(9, 16)]):
    return a.mm(b)
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1003"]
    assert errors
    diag = errors[0]
    assert diag.expected is not None and "rank-2" in diag.expected
    assert diag.actual is not None and "[4, 8]" in diag.actual


def test_tsf1003_einsum_structured_fields() -> None:
    """einsum with mismatched contraction dims carries structured fields."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(a: Annotated[torch.Tensor, Shape(3, 4)],
       b: Annotated[torch.Tensor, Shape(5, 6)]):
    return torch.einsum("ij,jk->ik", a, b)
"""
    report = analyze_source(source, Path("m.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1003"]
    assert errors
    diag = errors[0]
    assert diag.expected is not None and "ij,jk->ik" in diag.expected
    assert diag.actual is not None and "[3, 4]" in diag.actual and "[5, 6]" in diag.actual


def test_linear_non_literal_in_features_does_not_error() -> None:
    """When nn.Linear's in_features is a variable (None spec), infer succeeds, no TSF1007."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape


class M(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 999)]):
        return self.fc(x)
"""
    report = analyze_source(source, Path("m.py"))
    # Linear with a variable in_features cannot be checked against the input's
    # last dim; inference must still succeed and no TSF1007 may fire.
    assert not any(d.code == "TSF1007" for d in report.diagnostics)


# ---------------------------------------------------------------------------
# Feature: open-ended slices in analyzer
# ---------------------------------------------------------------------------


def test_open_ended_slice_constant_dim() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def f(x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]):
    y = x[:, 1:]
    return y
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "y"), None)
    assert hover is not None
    assert hover.shape == "[B, 2, 32, 32]"


# ---------------------------------------------------------------------------
# Feature: reshape symbolic cancellation in analyzer
# ---------------------------------------------------------------------------


def test_reshape_symbolic_cancel_in_analyzer() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def f(x: Annotated[torch.Tensor, Shape("B", 3, 4)]):
    y = x.reshape(x.shape[0], 6, -1)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "y"), None)
    assert hover is not None
    assert hover.shape == "[B, 6, 2]"


# ---------------------------------------------------------------------------
# TSF1012 — symbolic dim used where constant is required
# ---------------------------------------------------------------------------


def test_tsf1012_linear_symbolic_last_dim() -> None:
    """TSF1012 warning when symbolic dim is passed as in_features to nn.Linear."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(768, 256)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", "D")]):
        y = self.fc(x)
"""
    report = analyze_source(source, Path("m.py"))
    warnings = [d for d in report.diagnostics if d.code == "TSF1012"]
    assert len(warnings) == 1
    assert "D" in warnings[0].message
    assert "768" in warnings[0].message
    assert warnings[0].severity == "warning"
    # Inference still produces a result despite the warning.
    assert any(hover.name == "y" and hover.shape == "[B, T, 256]" for hover in report.hovers)


def test_tsf1012_conv2d_symbolic_channel_dim() -> None:
    """TSF1012 warning when symbolic channel dim is passed as in_channels to nn.Conv2d."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "C", "H", "W")]):
        y = self.conv(x)
"""
    report = analyze_source(source, Path("m.py"))
    warnings = [d for d in report.diagnostics if d.code == "TSF1012"]
    assert len(warnings) == 1
    assert "C" in warnings[0].message
    assert "3" in warnings[0].message
    assert warnings[0].severity == "warning"
    # Inference still produces a result despite the warning (no TSF1007 mismatch error).
    assert not any(d.code == "TSF1007" for d in report.diagnostics)
    assert any(hover.name == "y" and "16" in hover.shape for hover in report.hovers)


def test_tsf1012_no_warning_when_constant_matches() -> None:
    """No TSF1012 when the constant dim matches the required value."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 3, "H", "W")]):
        y = self.conv(x)
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.code == "TSF1012" for d in report.diagnostics)


def test_tsf1012_no_warning_in_unannotated_function() -> None:
    """No TSF1012 outside of annotated functions (no active shape contract)."""
    source = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(768, 256)

    def forward(self, x):
        y = self.fc(x)
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.code == "TSF1012" for d in report.diagnostics)

from __future__ import annotations

from pathlib import Path

from torchshapeflow.analyzer import analyze_source

# ---------------------------------------------------------------------------
# nn.MultiheadAttention
# ---------------------------------------------------------------------------


def test_multihead_attention_output_shape() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.attn = nn.MultiheadAttention(64, 8, batch_first=True)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
        out, _ = self.attn(x, x, x)
"""
    report = analyze_source(source, Path("mha.py"))
    assert report.diagnostics == []
    assert any(hover.name == "out" and hover.shape == "[B, T, 64]" for hover in report.hovers)


def test_multihead_attention_chained() -> None:
    """Output of MHA fed into a Linear layer."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.attn = nn.MultiheadAttention(64, 8, batch_first=True)
        self.proj = nn.Linear(64, 32)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
        out, _ = self.attn(x, x, x)
        y = self.proj(out)
"""
    report = analyze_source(source, Path("mha.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, T, 32]" for hover in report.hovers)


# ---------------------------------------------------------------------------
# nn.LSTM
# ---------------------------------------------------------------------------


def test_lstm_flat_unpack() -> None:
    """Top-level unpack: out, state = lstm(x), then subscript the nested state tuple."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", "N", 128)]):
        out, state = self.encoder(x)
        h_n = state[0]
        c_n = state[1]
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.severity == "error" for d in report.diagnostics)
    assert any(hover.name == "out" and hover.shape == "[L, N, 256]" for hover in report.hovers)
    assert any(hover.name == "h_n" and hover.shape == "[1, N, 256]" for hover in report.hovers)
    assert any(hover.name == "c_n" and hover.shape == "[1, N, 256]" for hover in report.hovers)


def test_lstm_nested_unpack() -> None:
    """Nested unpack: out, (h, c) = lstm(x)."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", "N", 128)]):
        out, (h, c) = self.encoder(x)
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.severity == "error" for d in report.diagnostics)
    assert any(hover.name == "out" and hover.shape == "[L, N, 256]" for hover in report.hovers)
    assert any(hover.name == "h" and hover.shape == "[1, N, 256]" for hover in report.hovers)
    assert any(hover.name == "c" and hover.shape == "[1, N, 256]" for hover in report.hovers)


def test_lstm_nested_discard_pattern() -> None:
    """Common pattern: _, (hidden, _) = lstm(x)."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256, num_layers=2)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", "N", 128)]):
        _, (hidden, _) = self.encoder(x)
        final = hidden[-1]
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.severity == "error" for d in report.diagnostics)
    assert any(hover.name == "hidden" and hover.shape == "[2, N, 256]" for hover in report.hovers)


def test_lstm_batch_first() -> None:
    """batch_first=True swaps L and N in output."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256, batch_first=True)

    def forward(self, x: Annotated[torch.Tensor, Shape("N", "L", 128)]):
        out, (h, c) = self.encoder(x)
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.severity == "error" for d in report.diagnostics)
    assert any(hover.name == "out" and hover.shape == "[N, L, 256]" for hover in report.hovers)
    assert any(hover.name == "h" and hover.shape == "[1, N, 256]" for hover in report.hovers)


def test_lstm_bidirectional() -> None:
    """bidirectional=True doubles output hidden dim and h_n first dim."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256, bidirectional=True)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", "N", 128)]):
        out, (h, c) = self.encoder(x)
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.severity == "error" for d in report.diagnostics)
    assert any(hover.name == "out" and hover.shape == "[L, N, 512]" for hover in report.hovers)
    assert any(hover.name == "h" and hover.shape == "[2, N, 256]" for hover in report.hovers)


def test_lstm_chained_linear() -> None:
    """LSTM output fed into a Linear layer."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256, num_layers=2, batch_first=True)
        self.head = nn.Linear(256, 10)

    def forward(self, x: Annotated[torch.Tensor, Shape("N", "L", 128)]):
        _, (hidden, _) = self.encoder(x)
        final_hidden = hidden[-1]
        return self.head(final_hidden)
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.severity == "error" for d in report.diagnostics)
    assert any(hover.name == "hidden" and hover.shape == "[2, N, 256]" for hover in report.hovers)


def test_lstm_state_subscript_preserves_nested_tuple_structure() -> None:
    """Indexing lstm(x)[1][0] should recover h_n, not flatten the state tuple."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", "N", 128)]):
        state = self.encoder(x)[1]
        h = state[0]
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.severity == "error" for d in report.diagnostics)
    assert not any(hover.name == "state" for hover in report.hovers)
    assert any(hover.name == "h" and hover.shape == "[1, N, 256]" for hover in report.hovers)


def test_lstm_input_size_mismatch_reports_tsf1007() -> None:
    """Definite trailing-dim mismatches should fail for nn.LSTM."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", "N", 64)]):
        out, _ = self.encoder(x)
"""
    report = analyze_source(source, Path("m.py"))
    assert any(d.code == "TSF1007" and "nn.LSTM" in d.message for d in report.diagnostics)


def test_lstm_symbolic_input_size_emits_tsf1012() -> None:
    """Symbolic trailing dims should warn when nn.LSTM expects a literal input_size."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", "N", "D")]):
        out, _ = self.encoder(x)
"""
    report = analyze_source(source, Path("m.py"))
    warnings = [d for d in report.diagnostics if d.code == "TSF1012"]
    assert len(warnings) == 1
    assert "nn.LSTM expects input_size=128" in warnings[0].message
    assert any(hover.name == "out" and hover.shape == "[L, N, 256]" for hover in report.hovers)


def test_lstm_proj_size_shapes() -> None:
    """proj_size changes output and h_n while c_n keeps hidden_size."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.encoder = nn.LSTM(128, 256, proj_size=64)

    def forward(self, x: Annotated[torch.Tensor, Shape("L", "N", 128)]):
        out, (h, c) = self.encoder(x)
"""
    report = analyze_source(source, Path("m.py"))
    assert not any(d.severity == "error" for d in report.diagnostics)
    assert any(hover.name == "out" and hover.shape == "[L, N, 64]" for hover in report.hovers)
    assert any(hover.name == "h" and hover.shape == "[1, N, 64]" for hover in report.hovers)
    assert any(hover.name == "c" and hover.shape == "[1, N, 256]" for hover in report.hovers)


def test_tensor_method_mm() -> None:
    """x.mm(y) tensor method form should work like torch.mm."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape(4, 8)],
       y: Annotated[torch.Tensor, Shape(8, 16)]):
    z = x.mm(y)
"""
    report = analyze_source(source, Path("mm.py"))
    assert report.diagnostics == []
    assert any(hover.name == "z" and hover.shape == "[4, 16]" for hover in report.hovers)


def test_einsum_dim_mismatch_emits_tsf1003() -> None:
    """einsum with mismatched contraction dims should emit TSF1003."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(q: Annotated[torch.Tensor, Shape("B", "T", 8)],
       k: Annotated[torch.Tensor, Shape("B", 9, "T")]):
    out = torch.einsum("bik,bkj->bij", q, k)
"""
    report = analyze_source(source, Path("ein.py"))
    assert any(d.code == "TSF1003" for d in report.diagnostics)


def test_f_interpolate_size_tuple() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn.functional as F
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C", 32, 32)]):
    y = F.interpolate(x, size=(64, 64), mode="bilinear")
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, C, 64, 64]" for hover in report.hovers)


def test_f_interpolate_size_variable() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn.functional as F
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C", 16, 16)],
       labels: Annotated[torch.Tensor, Shape("B", 32, 32)]):
    y = F.interpolate(x, size=labels.shape[-2:], mode="bilinear")
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, C, 32, 32]" for hover in report.hovers)


def test_f_interpolate_scale_factor() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn.functional as F
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C", 16, 16)]):
    y = F.interpolate(x, scale_factor=2.0)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, C, 32, 32]" for hover in report.hovers)


def test_argmax_reduction() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", 64, "H", "W")]):
    y = x.argmax(dim=1)
    z = torch.argmax(x, dim=1)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, H, W]" for hover in report.hovers)
    assert any(hover.name == "z" and hover.shape == "[B, H, W]" for hover in report.hovers)


def test_nanmean_reduction() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
    y = torch.nanmean(x, dim=-1)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, T]" for hover in report.hovers)


def test_torch_flip() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C", "H", "W")]):
    y = torch.flip(x, dims=[3])
    z = x.flip(dims=[2, 3])
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, C, H, W]" for hover in report.hovers)
    assert any(hover.name == "z" and hover.shape == "[B, C, H, W]" for hover in report.hovers)


def test_f_one_hot() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn.functional as F
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "H", "W")]):
    y = F.one_hot(x, num_classes=64)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, H, W, 64]" for hover in report.hovers)


def test_torch_topk() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", 256)]):
    top = torch.topk(x, k=10, dim=-1).values
    y = top.mean(dim=-1)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B]" for hover in report.hovers)


def test_diagonal() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", 64, 64)]):
    y = x.diagonal(dim1=-2, dim2=-1)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, 64]" for hover in report.hovers)


def test_index_select() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", 64, "H")],
       idx: Annotated[torch.Tensor, Shape(10)]):
    y = x.index_select(1, idx)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, 10, H]" for hover in report.hovers)


def test_torch_isfinite_passthrough() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T")]):
    y = torch.isfinite(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, T]" for hover in report.hovers)


def test_bincount_unknown_shape() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape(100)]):
    y = torch.bincount(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[?]" for hover in report.hovers)


def test_reshape_with_runtime_int_parameter() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C", "H", "W")], num_classes: int = 64):
    flat = x.reshape(-1)
    out = flat.reshape(num_classes, num_classes)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(
        hover.name == "out" and hover.shape == "[num_classes, num_classes]"
        for hover in report.hovers
    )


def test_reshape_with_batch_and_runtime_dim() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "N")], num_classes: int = 10):
    y = torch.bincount(x.reshape(-1), minlength=num_classes * num_classes)
    z = y.reshape(num_classes, num_classes)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(
        hover.name == "z" and hover.shape == "[num_classes, num_classes]" for hover in report.hovers
    )


def test_function_signature_hover() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def attention_scores(
    q: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
    k: Annotated[torch.Tensor, Shape("B", "H", "T", "D")],
) -> Annotated[torch.Tensor, Shape("B", "H", "T", "T")]:
    return q.matmul(k.transpose(-2, -1))
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    sig = next(
        (h for h in report.hovers if h.name == "attention_scores"),
        None,
    )
    assert sig is not None
    assert sig.shape == "(\n  q: [B, H, T, D],\n  k: [B, H, T, D]\n) → [B, H, T, T]"
    assert sig.kind == "signature"


def test_module_typealias_hover() -> None:
    source = """
from typing import Annotated, TypeAlias
import torch
from torchshapeflow import Shape

Batch: TypeAlias = Annotated[torch.Tensor, Shape("B", "T", 64)]
"""
    report = analyze_source(source, Path("f.py"))
    hover = next((h for h in report.hovers if h.name == "Batch"), None)
    assert hover is not None
    assert hover.shape == "[B, T, 64]"
    assert hover.kind == "alias"


def test_alias_reference_hover_in_local_annotation() -> None:
    source = """
import torch

def fn():
    from typing import Annotated, TypeAlias
    Batch: TypeAlias = Annotated[torch.Tensor, "B T 64"]
    x: Batch = torch.load("batch.pt")
"""
    report = analyze_source(source, Path("f.py"))
    alias_hovers = [hover for hover in report.hovers if hover.name == "Batch"]
    assert alias_hovers
    assert all(hover.shape == "[B, T, 64]" for hover in alias_hovers)
    assert all(hover.kind == "alias" for hover in alias_hovers)

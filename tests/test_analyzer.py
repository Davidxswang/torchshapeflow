from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from torchshapeflow.analyzer import analyze_path, analyze_source
from torchshapeflow.index import ProjectIndex


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


# ---------------------------------------------------------------------------
# split / chunk
# ---------------------------------------------------------------------------


def test_chunk_tuple_unpack() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 192)]):
    q, k, v = x.chunk(3, dim=-1)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "q" and hover.shape == "[B, T, 64]" for hover in report.hovers)
    assert any(hover.name == "k" and hover.shape == "[B, T, 64]" for hover in report.hovers)
    assert any(hover.name == "v" and hover.shape == "[B, T, 64]" for hover in report.hovers)


def test_chunk_keyword_dim() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", 256)]):
    a, b = x.chunk(2, dim=-1)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "a" and hover.shape == "[B, 128]" for hover in report.hovers)
    assert any(hover.name == "b" and hover.shape == "[B, 128]" for hover in report.hovers)


def test_split_int_size_unpack() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 192)]):
    q, k, v = x.split(64, dim=-1)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "q" and hover.shape == "[B, T, 64]" for hover in report.hovers)
    assert any(hover.name == "k" and hover.shape == "[B, T, 64]" for hover in report.hovers)
    assert any(hover.name == "v" and hover.shape == "[B, T, 64]" for hover in report.hovers)


def test_split_list_sizes_unpack() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 10)]):
    a, b = x.split([3, 7], dim=-1)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "a" and hover.shape == "[B, T, 3]" for hover in report.hovers)
    assert any(hover.name == "b" and hover.shape == "[B, T, 7]" for hover in report.hovers)


def test_chunk_stored_in_env_subscript() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 128)]):
    chunks = x.chunk(2, dim=-1)
    first = chunks[0]
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "first" and hover.shape == "[B, T, 64]" for hover in report.hovers)


def test_torch_split_function() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 192)]):
    q, k, v = torch.split(x, 64, dim=-1)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "q" and hover.shape == "[B, T, 64]" for hover in report.hovers)
    assert any(hover.name == "v" and hover.shape == "[B, T, 64]" for hover in report.hovers)


def test_transformer_qkv_split() -> None:
    """Single-projection QKV split pattern common in efficient transformers."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Attention(nn.Module):
    def __init__(self):
        self.qkv = nn.Linear(64, 192)
        self.out = nn.Linear(64, 64)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        kt = k.transpose(-2, -1)
        scores = q @ kt
        attn = torch.softmax(scores.float(), dim=-1).half()
        out = attn @ v
        return self.out(out)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "q" and hover.shape == "[B, T, 64]" for hover in report.hovers)
    assert any(hover.name == "scores" and hover.shape == "[B, T, T]" for hover in report.hovers)
    assert any(hover.name == "out" and hover.shape == "[B, T, 64]" for hover in report.hovers)


# ---------------------------------------------------------------------------
# PassthroughSpec: BatchNorm2d, ReLU
# ---------------------------------------------------------------------------


def test_passthrough_batchnorm2d() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self):
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 8, 32, 32)]):
        return self.bn(x)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    # The return value should have the same shape as x.
    assert any(hover.shape == "[B, 8, 32, 32]" for hover in report.hovers)


def test_passthrough_relu() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self):
        self.act = nn.ReLU()

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "C", 16, 16)]):
        y = self.act(x)
        return y
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, C, 16, 16]" for hover in report.hovers)


# ---------------------------------------------------------------------------
# EmbeddingSpec
# ---------------------------------------------------------------------------


def test_embedding_shape() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.embed = nn.Embedding(1000, 64)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T")]):
        return self.embed(x)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    # embedding appends 64 to (B, T) → (B, T, 64)
    assert any(hover.shape == "[B, T, 64]" for hover in report.hovers)


# ---------------------------------------------------------------------------
# Pool2dSpec: MaxPool2d, AvgPool2d
# ---------------------------------------------------------------------------


def test_maxpool2d_shape() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self):
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "C", 8, 8)]):
        y = self.pool(x)
        return y
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, C, 4, 4]" for hover in report.hovers)


def test_avgpool2d_shape() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self):
        self.pool = nn.AvgPool2d(4, stride=2)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "C", 16, 16)]):
        y = self.pool(x)
        return y
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    # H_out = floor((16 + 0 - 4) / 2 + 1) = floor(12/2 + 1) = 7
    assert any(hover.name == "y" and hover.shape == "[B, C, 7, 7]" for hover in report.hovers)


# ---------------------------------------------------------------------------
# TSF1009: return shape mismatch
# ---------------------------------------------------------------------------


def test_return_shape_rank_mismatch_tsf1009() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 64)]) -> Annotated[torch.Tensor, Shape("B", 64)]:
    return x
"""
    report = analyze_source(source, Path("memory.py"))
    assert any(d.code == "TSF1009" for d in report.diagnostics)


# ---------------------------------------------------------------------------
# Reduction: torch.sum(x, dim=1)
# ---------------------------------------------------------------------------


def test_reduction_global_torch_sum() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
    y = torch.sum(x, dim=1)
    return y
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, 64]" for hover in report.hovers)


# ---------------------------------------------------------------------------
# Module alias: m = self.linear; m(x)
# ---------------------------------------------------------------------------


def test_module_alias_linear() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Net(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(64, 32)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
        m = self.linear
        y = m(x)
        return y
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, T, 32]" for hover in report.hovers)


# ---------------------------------------------------------------------------
# Augmented assignment (+=, -=, etc.)
# ---------------------------------------------------------------------------


def test_augmented_assignment_residual() -> None:
    """x += y should update x's shape via broadcast (residual connection pattern)."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 64)],
       r: Annotated[torch.Tensor, Shape("B", "T", 64)]):
    x += r
    y = x.transpose(1, 2)
"""
    report = analyze_source(source, Path("memory.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, 64, T]" for hover in report.hovers)


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


# ---------------------------------------------------------------------------
# nn.Sequential
# ---------------------------------------------------------------------------


def test_sequential_basic() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
        y = self.net(x)
"""
    report = analyze_source(source, Path("seq.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, T, 32]" for hover in report.hovers)


def test_sequential_multi_layer() -> None:
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

    def forward(self, x: Annotated[torch.Tensor, Shape("B", 128)]):
        y = self.net(x)
"""
    report = analyze_source(source, Path("seq.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, 16]" for hover in report.hovers)


# ---------------------------------------------------------------------------
# torch.einsum
# ---------------------------------------------------------------------------


def test_torch_einsum_bmm() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(q: Annotated[torch.Tensor, Shape("B", "T", "D")],
       k: Annotated[torch.Tensor, Shape("B", "D", "T")]):
    out = torch.einsum("bik,bkj->bij", q, k)
"""
    report = analyze_source(source, Path("ein.py"))
    assert report.diagnostics == []
    assert any(hover.name == "out" and hover.shape == "[B, T, T]" for hover in report.hovers)


def test_torch_einsum_matrix_vector() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(A: Annotated[torch.Tensor, Shape(4, 8)],
       v: Annotated[torch.Tensor, Shape(8)]):
    out = torch.einsum("ij,j->i", A, v)
"""
    report = analyze_source(source, Path("ein.py"))
    assert report.diagnostics == []
    assert any(hover.name == "out" and hover.shape == "[4]" for hover in report.hovers)


def test_torch_einsum_list_form() -> None:
    # torch.einsum("ij,jk->ik", [A, B])
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(A: Annotated[torch.Tensor, Shape("M", "K")],
       B: Annotated[torch.Tensor, Shape("K", "N")]):
    out = torch.einsum("ij,jk->ik", [A, B])
"""
    report = analyze_source(source, Path("ein.py"))
    assert report.diagnostics == []
    assert any(hover.name == "out" and hover.shape == "[M, N]" for hover in report.hovers)


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
# TSF2001: unsupported tensor method warning
# ---------------------------------------------------------------------------


def test_tsf2001_unsupported_tensor_method() -> None:
    """Annotated function calling x.some_weird_method() should produce TSF2001."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
    y = x.some_weird_method()
    return y
"""
    report = analyze_source(source, Path("f.py"))
    warnings = [d for d in report.diagnostics if d.code == "TSF2001"]
    assert len(warnings) == 1
    assert ".some_weird_method" in warnings[0].message
    assert warnings[0].severity == "warning"


def test_tsf2001_no_false_positive_known_ops() -> None:
    """Known operations like reshape, flatten should NOT produce TSF2001."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)]):
    y = x.reshape(x.shape[0], -1)
    z = y.flatten(0)
    w = z.detach()
    v = w.contiguous()
    return v
"""
    report = analyze_source(source, Path("f.py"))
    tsf2001 = [d for d in report.diagnostics if d.code == "TSF2001"]
    assert len(tsf2001) == 0


def test_tsf2001_no_warning_in_unannotated_function() -> None:
    """Unannotated function should NOT produce TSF2001."""
    source = """
import torch

def fn(x):
    y = x.some_weird_method()
    return y
"""
    report = analyze_source(source, Path("f.py"))
    tsf2001 = [d for d in report.diagnostics if d.code == "TSF2001"]
    assert len(tsf2001) == 0


def test_tsf2001_silent_for_non_tensor_methods() -> None:
    """Methods like .item(), .numpy(), .tolist(), .dim() should NOT produce TSF2001."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
    a = x.item()
    b = x.numpy()
    c = x.tolist()
    d = x.dim()
"""
    report = analyze_source(source, Path("f.py"))
    tsf2001 = [d for d in report.diagnostics if d.code == "TSF2001"]
    assert len(tsf2001) == 0


# ---------------------------------------------------------------------------
# TSF2002: call to unannotated function with tensor arg
# ---------------------------------------------------------------------------


def test_tsf2002_unannotated_function_call() -> None:
    """Annotated function calling helper(x) where helper has no annotations."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def helper(x):
    return x

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
    y = helper(x)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    warnings = [d for d in report.diagnostics if d.code == "TSF2002"]
    assert len(warnings) == 1
    assert "helper" in warnings[0].message
    assert warnings[0].severity == "warning"


def test_tsf2002_no_false_positive_builtins() -> None:
    """Calling print(x), len(x) should NOT produce TSF2002."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
    print(x)
    n = len(x)
"""
    report = analyze_source(source, Path("f.py"))
    tsf2002 = [d for d in report.diagnostics if d.code == "TSF2002"]
    assert len(tsf2002) == 0


def test_tsf2002_no_warning_in_unannotated_function() -> None:
    """Unannotated function should NOT produce TSF2002."""
    source = """
import torch

def helper(x):
    return x

def fn(x):
    y = helper(x)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    tsf2002 = [d for d in report.diagnostics if d.code == "TSF2002"]
    assert len(tsf2002) == 0


def test_tsf2002_no_warning_for_annotated_function() -> None:
    """Calling a function that has Shape annotations should NOT produce TSF2002."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def helper(
    x: Annotated[torch.Tensor, Shape("B", "D")],
) -> Annotated[torch.Tensor, Shape("B", "D")]:
    return x

def fn(a: Annotated[torch.Tensor, Shape(8, 64)]):
    out = helper(a)
"""
    report = analyze_source(source, Path("f.py"))
    tsf2002 = [d for d in report.diagnostics if d.code == "TSF2002"]
    assert len(tsf2002) == 0


# ---------------------------------------------------------------------------
# TSF2003: unresolvable self.xxx module
# ---------------------------------------------------------------------------


def test_tsf2003_unresolvable_self_module() -> None:
    """Class with self.custom = SomeModule() and self.custom(x) in annotated forward."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.custom = SomeUnknownModule()

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
        y = self.custom(x)
        return y
"""
    report = analyze_source(source, Path("f.py"))
    warnings = [d for d in report.diagnostics if d.code == "TSF2003"]
    assert len(warnings) == 1
    assert "self.custom" in warnings[0].message
    assert warnings[0].severity == "warning"


def test_tsf2003_no_warning_for_known_modules() -> None:
    """Known modules (nn.Linear, nn.Conv2d, etc.) should NOT produce TSF2003."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(64, 32)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
        y = self.linear(x)
        return y
"""
    report = analyze_source(source, Path("f.py"))
    tsf2003 = [d for d in report.diagnostics if d.code == "TSF2003"]
    assert len(tsf2003) == 0


def test_local_typealias_and_annotated_variable_seed_inference() -> None:
    source = """
import torch

def fn():
    from typing import Annotated, TypeAlias
    Batch: TypeAlias = Annotated[torch.Tensor, "B T 64"]
    x: Batch = torch.load("batch.pt")
    y = x.transpose(-2, -1)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "x" and hover.shape == "[B, T, 64]" for hover in report.hovers)
    assert any(hover.name == "y" and hover.shape == "[B, 64, T]" for hover in report.hovers)


def test_local_plain_alias_assignment_is_supported() -> None:
    source = """
import torch

def fn():
    from typing import Annotated
    Batch = Annotated[torch.Tensor, "B T 64"]
    x: Batch = torch.load("batch.pt")
    y = x.transpose(-2, -1)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, 64, T]" for hover in report.hovers)


def test_local_annotated_variable_emits_mismatch() -> None:
    source = """
from typing import Annotated
import torch

def fn(x: Annotated[torch.Tensor, "B 64"]):
    y: Annotated[torch.Tensor, "B 32"] = x
"""
    report = analyze_source(source, Path("f.py"))
    errors = [diagnostic for diagnostic in report.diagnostics if diagnostic.code == "TSF1011"]
    assert len(errors) == 1
    assert "[B, 64]" in errors[0].message
    assert "[B, 32]" in errors[0].message


def test_tsf2003_no_warning_in_unannotated_forward() -> None:
    """Unannotated forward method should NOT produce TSF2003."""
    source = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        self.custom = SomeUnknownModule()

    def forward(self, x):
        y = self.custom(x)
        return y
"""
    report = analyze_source(source, Path("f.py"))
    tsf2003 = [d for d in report.diagnostics if d.code == "TSF2003"]
    assert len(tsf2003) == 0


# ---------------------------------------------------------------------------
# Missing error/warning diagnostics — operator argument resolution
# ---------------------------------------------------------------------------


def test_permute_non_literal_args_warns() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C", "H", "W")], order):
    y = x.permute(order)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    warns = [d for d in report.diagnostics if d.code == "TSF2001"]
    assert len(warns) == 1
    assert "permute" in warns[0].message


def test_permute_non_literal_no_warn_unannotated() -> None:
    source = """
import torch

def fn(x, order):
    y = x.permute(order)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    assert len(report.diagnostics) == 0


def test_transpose_non_literal_args_warns() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C", "H")], d):
    y = x.transpose(d, 0)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    warns = [d for d in report.diagnostics if d.code == "TSF2001"]
    assert len(warns) == 1
    assert "transpose" in warns[0].message


def test_flatten_non_literal_args_warns() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C", "H", "W")], s):
    y = x.flatten(s)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    warns = [d for d in report.diagnostics if d.code == "TSF2001"]
    assert len(warns) == 1
    assert "flatten" in warns[0].message


def test_unsqueeze_non_literal_arg_warns() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "C")], d):
    y = x.unsqueeze(d)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    warns = [d for d in report.diagnostics if d.code == "TSF2001"]
    assert len(warns) == 1
    assert "unsqueeze" in warns[0].message


def test_chunk_bad_dim_errors() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T")]):
    a, b = x.chunk(2, dim=5)
    return a
"""
    report = analyze_source(source, Path("f.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1008"]
    assert len(errors) == 1
    assert "chunk" in errors[0].message


def test_movedim_bad_indices_errors() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T")]):
    y = x.movedim(5, 0)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1008"]
    assert len(errors) == 1
    assert "movedim" in errors[0].message


def test_movedim_functional_bad_indices_errors() -> None:
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape

def fn(x: Annotated[torch.Tensor, Shape("B", "T")]):
    y = torch.movedim(x, 5, 0)
    return y
"""
    report = analyze_source(source, Path("f.py"))
    errors = [d for d in report.diagnostics if d.code == "TSF1008"]
    assert len(errors) == 1
    assert "movedim" in errors[0].message


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


# ---------------------------------------------------------------------------
# Feature: self.attr scalar tracking from __init__
# ---------------------------------------------------------------------------


def test_self_attr_in_reshape() -> None:
    """self.xxx = init_param in __init__ makes self.xxx usable as reshape dim."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self, horizon: int, counties: int):
        self.horizon = horizon
        self.counties = counties

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "horizon_counties")]):
        batch_size, _ = x.shape
        return x.reshape(batch_size, self.horizon, self.counties)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "<return>"), None)
    assert hover is not None
    assert hover.shape == "[B, horizon, counties]"


def test_binop_out_features_in_linear() -> None:
    """nn.Linear with a Name*Name output dim captures it as an expression."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self, hidden: int, horizon: int, counties: int):
        self.proj = nn.Linear(hidden, horizon * counties)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "hidden")]):
        return self.proj(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "<return>"), None)
    assert hover is not None
    assert hover.shape == "[B, horizon*counties]"


def test_self_attr_and_binop_combined() -> None:
    """LSTM → Linear(binop) → reshape(self.attr) full trace."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn
from torchshapeflow import Shape

class Model(nn.Module):
    def __init__(self, hidden: int, horizon: int, counties: int):
        self.horizon = horizon
        self.counties = counties
        self.encoder = nn.LSTM(input_size=64, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon * counties)

    def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 64)]):
        _, (h, _) = self.encoder(x)
        final = h[-1]
        out = self.head(final)
        batch_size, _ = out.shape
        return out.reshape(batch_size, self.horizon, self.counties)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    hover = next((h for h in report.hovers if h.name == "<return>"), None)
    assert hover is not None
    assert hover.shape == "[B, horizon, counties]"


def test_init_tensor_param_hover() -> None:
    """Annotated tensor params in __init__ should produce hovers and a signature."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, adjacency: Annotated[torch.Tensor, "N N"]):
        self.register_buffer("adjacency", adjacency)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    adjacency_hover = next((h for h in report.hovers if h.name == "adjacency"), None)
    assert adjacency_hover is not None
    assert adjacency_hover.shape == "[N, N]"
    signature_hover = next((h for h in report.hovers if h.name == "__init__"), None)
    assert signature_hover is not None
    assert signature_hover.shape == "(adjacency: [N, N])"
    assert signature_hover.kind == "signature"


def test_register_buffer_tracks_tensor_self_attr() -> None:
    """register_buffer(name, tensor) should make self.name available as a tensor later."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, adjacency: Annotated[torch.Tensor, "N N"]):
        self.register_buffer("adjacency", adjacency.to(dtype=torch.float32))

    def forward(self, x: Annotated[torch.Tensor, "B N F"]):
        a = self.adjacency
        y = torch.matmul(self.adjacency, x)
        return y
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    adjacency_hovers = [h for h in report.hovers if h.name == "adjacency"]
    assert adjacency_hovers
    assert any(h.shape == "[N, N]" for h in adjacency_hovers)
    y_hover = next((h for h in report.hovers if h.name == "y"), None)
    assert y_hover is not None
    assert y_hover.shape == "[B, N, F]"


def test_direct_tensor_self_attr_tracks_later_use() -> None:
    """self.attr = tensor_expr in __init__ should make self.attr available later."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, adjacency: Annotated[torch.Tensor, "N N"]):
        self.adjacency = adjacency.to(dtype=torch.float32)

    def forward(self, x: Annotated[torch.Tensor, "B N F"]):
        y = torch.matmul(self.adjacency, x)
        return y
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    assert any(h.name == "adjacency" and h.shape == "[N, N]" for h in report.hovers)
    y_hover = next((h for h in report.hovers if h.name == "y"), None)
    assert y_hover is not None
    assert y_hover.shape == "[B, N, F]"


def test_loop_built_sequential_from_positive_depth() -> None:
    """A loop-built Sequential with a positive symbolic depth should be summarized."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, depth: int):
        if depth <= 0:
            raise ValueError("depth must be positive")
        layers = []
        for layer_idx in range(depth):
            in_dim = 64 if layer_idx == 0 else 32
            layers.append(nn.Linear(in_dim, 32))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Annotated[torch.Tensor, "B 64"]):
        y = self.net(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    y_hover = next((h for h in report.hovers if h.name == "y"), None)
    assert y_hover is not None
    assert y_hover.shape == "[B, 32]"


def test_loop_built_sequential_from_annotated_empty_list() -> None:
    """Annotated empty-list initialization should still seed the loop-built Sequential summary."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, depth: int):
        if depth <= 0:
            raise ValueError("depth must be positive")
        layers: list[nn.Module] = []
        for layer_idx in range(depth):
            in_dim = 64 if layer_idx == 0 else 32
            layers.append(nn.Linear(in_dim, 32))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Annotated[torch.Tensor, "B 64"]):
        y = self.net(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    y_hover = next((h for h in report.hovers if h.name == "y"), None)
    assert y_hover is not None
    assert y_hover.shape == "[B, 32]"


def test_unresolved_starred_sequential_does_not_fall_back_to_identity() -> None:
    """Unknown starred Sequential contents should lose inference rather than act like identity."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn

def build_layers():
    return []

class Model(nn.Module):
    def __init__(self):
        layers = build_layers()
        self.net = nn.Sequential(*layers)

    def forward(self, x: Annotated[torch.Tensor, "B 64"]):
        y = self.net(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert any(d.code == "TSF2003" and "self.net" in d.message for d in report.diagnostics)
    assert not any(h.name == "y" for h in report.hovers)


def test_custom_module_forward_signature_used_for_self_call() -> None:
    """Annotated custom nn.Module.forward should seed the spec for self.block(x)."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn

class GraphLayer(nn.Module):
    def __init__(self, input_feature_dim: int, hidden_dim: int):
        self.linear = nn.Linear(input_feature_dim, hidden_dim)

    def forward(
        self,
        x: Annotated[torch.Tensor, "B N input_feature_dim"],
    ) -> Annotated[torch.Tensor, "B N hidden_dim"]:
        return self.linear(x)

class Model(nn.Module):
    def __init__(self):
        self.block = GraphLayer(64, 32)

    def forward(self, x: Annotated[torch.Tensor, "B N 64"]):
        y = self.block(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    y_hover = next((h for h in report.hovers if h.name == "y"), None)
    assert y_hover is not None
    assert y_hover.shape == "[B, N, 32]"


def test_imported_custom_module_forward_signature_used_for_self_call() -> None:
    """Project-local imported custom modules should contribute a forward shape contract."""
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "blocks.py").write_text(
            """
from typing import Annotated
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: Annotated[torch.Tensor, "B in_dim"],
    ) -> Annotated[torch.Tensor, "B out_dim"]:
        return self.linear(x)
""",
            encoding="utf-8",
        )
        (root / "model.py").write_text(
            """
from typing import Annotated
import torch
import torch.nn as nn
from blocks import Block

class Model(nn.Module):
    def __init__(self):
        self.block = Block(64, 32)

    def forward(self, x: Annotated[torch.Tensor, "B 64"]):
        y = self.block(x)
""",
            encoding="utf-8",
        )
        report = analyze_path(root / "model.py", ProjectIndex())
    assert report.diagnostics == []
    y_hover = next((h for h in report.hovers if h.name == "y"), None)
    assert y_hover is not None
    assert y_hover.shape == "[B, 32]"


def test_imported_custom_modules_in_loop_built_sequential() -> None:
    """Imported project-local custom modules should work inside loop-built Sequential."""
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "blocks.py").write_text(
            """
from typing import Annotated
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: Annotated[torch.Tensor, "B in_dim"],
    ) -> Annotated[torch.Tensor, "B out_dim"]:
        return self.linear(x)
""",
            encoding="utf-8",
        )
        (root / "model.py").write_text(
            """
from typing import Annotated
import torch
import torch.nn as nn
from blocks import Block

class Model(nn.Module):
    def __init__(self, depth: int, hidden_dim: int = 32):
        if depth <= 0:
            raise ValueError("depth must be positive")
        layers: list[nn.Module] = []
        for layer_idx in range(depth):
            in_dim = 64 if layer_idx == 0 else hidden_dim
            layers.append(Block(in_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Annotated[torch.Tensor, "B 64"]):
        y = self.net(x)
""",
            encoding="utf-8",
        )
        report = analyze_path(root / "model.py", ProjectIndex())
    assert report.diagnostics == []
    y_hover = next((h for h in report.hovers if h.name == "y"), None)
    assert y_hover is not None
    assert y_hover.shape == "[B, hidden_dim]"


def test_loop_built_sequential_of_custom_modules() -> None:
    """Loop-built Sequential should preserve custom module specs when forward is annotated."""
    source = """
from typing import Annotated
import torch
import torch.nn as nn

class GraphLayer(nn.Module):
    def __init__(self, input_feature_dim: int, hidden_dim: int, dropout: float = 0.1):
        self.linear = nn.Linear(input_feature_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Annotated[torch.Tensor, "B N input_feature_dim"],
    ) -> Annotated[torch.Tensor, "B N hidden_dim"]:
        return self.dropout(self.activation(self.linear(x)))

class Model(nn.Module):
    def __init__(self, depth: int, hidden_dim: int = 32):
        if depth <= 0:
            raise ValueError("depth must be positive")
        layers = []
        for layer_idx in range(depth):
            in_dim = 64 if layer_idx == 0 else hidden_dim
            layers.append(GraphLayer(in_dim, hidden_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Annotated[torch.Tensor, "B N 64"]):
        y = self.net(x)
"""
    report = analyze_source(source, Path("f.py"))
    assert report.diagnostics == []
    y_hover = next((h for h in report.hovers if h.name == "y"), None)
    assert y_hover is not None
    assert y_hover.shape == "[B, N, hidden_dim]"

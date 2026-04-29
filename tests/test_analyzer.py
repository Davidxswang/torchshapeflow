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

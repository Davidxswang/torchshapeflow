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


def test_init_param_default_resolves_linear() -> None:
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
    hover = next(
        (
            h
            for h in report.hovers
            if h.name == "<return>" or (h.name != "x" and h.shape.startswith("["))
        ),
        None,
    )
    assert hover is not None
    assert hover.shape == "[B, 32]"


def test_init_param_no_default_drops_spec() -> None:
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
    # out_dim has no default, so linear spec is dropped — no hover for return.
    return_hovers = [h for h in report.hovers if h.name == "<return>"]
    assert len(return_hovers) == 0


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
    assert hover.shape == "[B, 128]"


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

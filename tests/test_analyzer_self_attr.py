from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from torchshapeflow.analyzer import analyze_path, analyze_source
from torchshapeflow.index import ProjectIndex

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

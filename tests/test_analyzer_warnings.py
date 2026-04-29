from __future__ import annotations

from pathlib import Path

from torchshapeflow.analyzer import analyze_source

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

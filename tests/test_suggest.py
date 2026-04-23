from __future__ import annotations

from pathlib import Path

from torchshapeflow.analyzer import analyze_source


def test_suggest_return_annotation_symbolic_shape() -> None:
    """Params annotated, no return annotation, body tracks — suggestion emitted.

    Annotation uses ``ast.unparse`` output (single-quoted strings) so that
    downstream tooling gets a parse-stable rendering.
    """
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
    assert sug.annotation == "Annotated[torch.Tensor, Shape('B', 'H', 'T', 'T')]"


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
    assert suggestions[0].annotation == "Annotated[torch.Tensor, Shape('B', 'T', 256)]"


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
    assert sug_payload["annotation"] == "Annotated[torch.Tensor, Shape('B')]"
    report_payload = report.to_dict()
    assert "suggestions" in report_payload
    assert report_payload["suggestions"] == [sug_payload]


# ---------------------------------------------------------------------------
# Control-flow safety (P1 review fix): only suggest when every path returns.


def test_no_suggestion_when_if_lacks_else_implicit_fallthrough() -> None:
    """An `if` without `else` lets the function fall through to None — no suggest."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)], flag: bool):
    if flag:
        return x
    # falls through — runtime returns None
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_no_suggestion_when_mixed_with_bare_return() -> None:
    """A bare `return` on any path means the function can return None."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)], flag: bool):
    if flag:
        return x
    return
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_suggest_when_if_else_both_branches_return() -> None:
    """Exhaustive if/else with tensor returns in both branches does suggest."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B", "T")], flag: bool):
    if flag:
        return x
    else:
        return x
"""
    report = analyze_source(source, Path("m.py"))
    assert len(report.suggestions) == 1
    assert report.suggestions[0].annotation == "Annotated[torch.Tensor, Shape('B', 'T')]"


def test_suggest_when_trailing_raise_backstops_return() -> None:
    """Function returns on some paths and raises on the rest — still safe to suggest."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)], flag: bool):
    if flag:
        return x
    raise ValueError("flag must be set")
"""
    report = analyze_source(source, Path("m.py"))
    assert len(report.suggestions) == 1


def test_no_suggestion_when_only_terminal_is_while_loop() -> None:
    """Loops aren't analyzed; don't guess whether they always return."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)]):
    while True:
        return x
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_no_suggestion_when_only_terminal_is_try_except() -> None:
    """try/except isn't analyzed for termination; skip."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)]):
    try:
        return x
    except Exception:
        return x
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


# ---------------------------------------------------------------------------
# Import-safe rendering (P2 review fix): reuse the caller's annotation spelling.


def test_suggestion_preserves_from_torch_import_tensor_spelling() -> None:
    """When the user wrote `Annotated[Tensor, ...]`, the suggestion keeps `Tensor`.

    Emitting `torch.Tensor` would reference a name the target file never imported.
    """
    source = """
from typing import Annotated
from torch import Tensor
from torchshapeflow import Shape


def fn(x: Annotated[Tensor, Shape("B", "T")]):
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert len(report.suggestions) == 1
    annotation = report.suggestions[0].annotation
    assert annotation == "Annotated[Tensor, Shape('B', 'T')]"
    # Defense-in-depth: the broken form must not appear.
    assert "torch.Tensor" not in annotation


def test_suggestion_preserves_qualified_annotated() -> None:
    """`typing.Annotated[...]` param → suggestion keeps the qualified spelling."""
    source = """
import typing
import torch
from torchshapeflow import Shape


def fn(x: typing.Annotated[torch.Tensor, Shape("B", "T")]):
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert len(report.suggestions) == 1
    assert report.suggestions[0].annotation == ("typing.Annotated[torch.Tensor, Shape('B', 'T')]")


def test_suggestion_preserves_string_shorthand_form() -> None:
    """`Annotated[Tensor, "B T"]` string shorthand → suggestion uses the same form."""
    source = """
from typing import Annotated
import torch


def fn(x: Annotated[torch.Tensor, "B T 768"]):
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert len(report.suggestions) == 1
    annotation = report.suggestions[0].annotation
    assert annotation == "Annotated[torch.Tensor, 'B T 768']"


def test_no_suggestion_when_param_uses_typealias() -> None:
    """A TypeAlias param (`x: Batch`) gives no inline template; skip.

    Building a suggestion would require expanding the alias to `Annotated[...]`
    with names that may or may not be in scope. Under-propose instead.
    """
    source = """
from typing import Annotated, TypeAlias
import torch
from torchshapeflow import Shape


Batch: TypeAlias = Annotated[torch.Tensor, Shape("B", "T")]


def fn(x: Batch):
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


# ---------------------------------------------------------------------------
# Generator guard (review fix): a yield makes the function return Generator[...].


def test_no_suggestion_for_generator_with_yield() -> None:
    """A `yield` in the body makes this a generator — never suggest a tensor return."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)]):
    yield x
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_no_suggestion_for_generator_with_yield_from() -> None:
    """`yield from` is equally generator-making."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)]):
    yield from [x, x]
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_no_suggestion_for_generator_with_trailing_return() -> None:
    """The specific case Codex flagged: yield + trailing `return X`.

    Without the generator guard, this satisfied every other precondition and
    would be proposed as `-> Annotated[Tensor, Shape('B')]` — which is false,
    because calling the function returns a Generator, not a tensor.
    """
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)]):
    yield x
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert report.suggestions == []


def test_suggestion_ignores_yield_inside_nested_def() -> None:
    """A yield inside a nested `def` makes that nested function a generator,
    not the outer one. The outer function is still a normal return and may
    receive a suggestion.
    """
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)]):
    def _inner():
        yield x
    return x
"""
    report = analyze_source(source, Path("m.py"))
    # The outer fn's return suggestion should still emit.
    outer_suggestions = [s for s in report.suggestions if s.function == "fn"]
    assert len(outer_suggestions) == 1
    assert outer_suggestions[0].annotation == "Annotated[torch.Tensor, Shape('B')]"


def test_suggestion_allows_generator_expression_in_body() -> None:
    """A generator expression `(x for x in ...)` does not use ast.Yield — it's
    a GeneratorExp literal. Functions using one are not generator functions."""
    source = """
from typing import Annotated
import torch
from torchshapeflow import Shape


def fn(x: Annotated[torch.Tensor, Shape("B",)]):
    _ = list(i for i in range(3))
    return x
"""
    report = analyze_source(source, Path("m.py"))
    assert len(report.suggestions) == 1
    assert report.suggestions[0].annotation == "Annotated[torch.Tensor, Shape('B')]"

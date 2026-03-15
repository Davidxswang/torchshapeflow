"""Tests for cross-file TypeAlias resolution and function-signature indexing."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from torchshapeflow.analyzer import analyze_source
from torchshapeflow.index import (
    ProjectIndex,
    apply_substitution,
    collect_imports,
    collect_raw_aliases,
    extract_func_sig,
    resolve_aliases,
    unify_dims,
)
from torchshapeflow.model import (
    ConstantDim,
    Dim,
    SymbolicDim,
    TensorShape,
)
from torchshapeflow.parser import parse_source

# ---------------------------------------------------------------------------
# unify_dims / apply_substitution
# ---------------------------------------------------------------------------


def test_unify_dims_basic() -> None:
    declared = (SymbolicDim("B"), SymbolicDim("T"), ConstantDim(768))
    actual = (ConstantDim(4), ConstantDim(16), ConstantDim(768))
    mapping = unify_dims(declared, actual)
    assert mapping == {"B": ConstantDim(4), "T": ConstantDim(16)}


def test_unify_dims_rank_mismatch_returns_empty() -> None:
    declared = (SymbolicDim("B"), SymbolicDim("T"))
    actual = (ConstantDim(4),)
    assert unify_dims(declared, actual) == {}


def test_unify_dims_constant_declared_skipped() -> None:
    # ConstantDim in declared is not mapped; only SymbolicDim entries are
    declared = (ConstantDim(3), SymbolicDim("C"))
    actual = (ConstantDim(3), ConstantDim(64))
    mapping = unify_dims(declared, actual)
    assert mapping == {"C": ConstantDim(64)}
    assert "3" not in mapping


def test_apply_substitution_replaces_symbolic() -> None:
    shape = TensorShape((SymbolicDim("B"), SymbolicDim("D"), ConstantDim(256)))
    mapping: dict[str, Dim] = {"B": ConstantDim(8), "D": ConstantDim(32)}
    result = apply_substitution(shape, mapping)
    assert str(result) == "[8, 32, 256]"


def test_apply_substitution_unknown_sym_stays() -> None:
    shape = TensorShape((SymbolicDim("B"), SymbolicDim("X")))
    mapping: dict[str, Dim] = {"B": ConstantDim(4)}
    result = apply_substitution(shape, mapping)
    assert str(result) == "[4, X]"


# ---------------------------------------------------------------------------
# collect_raw_aliases / collect_imports
# ---------------------------------------------------------------------------


def test_collect_raw_aliases_plain_assign() -> None:
    src = textwrap.dedent("""\
        from typing import Annotated
        import torch
        from torchshapeflow import Shape
        Image = Annotated[torch.Tensor, Shape(3, 224, 224)]
    """)
    module = parse_source(src)
    raw = collect_raw_aliases(module)
    assert "Image" in raw


def test_collect_raw_aliases_annotated_assign() -> None:
    src = textwrap.dedent("""\
        from typing import Annotated
        from typing_extensions import TypeAlias
        import torch
        from torchshapeflow import Shape
        Image: TypeAlias = Annotated[torch.Tensor, Shape(3, 224, 224)]
    """)
    module = parse_source(src)
    raw = collect_raw_aliases(module)
    assert "Image" in raw


def test_collect_raw_aliases_type_statement() -> None:
    if not hasattr(ast, "TypeAlias"):
        pytest.skip("Python type statements require an AST runtime with ast.TypeAlias.")
    src = textwrap.dedent("""\
        from typing import Annotated
        import torch
        from torchshapeflow import Shape
        type Image = Annotated[torch.Tensor, Shape(3, 224, 224)]
    """)
    module = parse_source(src)
    raw = collect_raw_aliases(module)
    assert "Image" in raw


def test_collect_imports_from_import() -> None:
    src = textwrap.dedent("""\
        from mymodule import Image, process
        from other import something as alias_name
    """)
    module = parse_source(src)
    imports = collect_imports(module)
    assert imports["Image"] == ("mymodule", "Image")
    assert imports["process"] == ("mymodule", "process")
    assert imports["alias_name"] == ("other", "something")


# ---------------------------------------------------------------------------
# resolve_aliases
# ---------------------------------------------------------------------------


def test_resolve_aliases_inline_annotated() -> None:
    src = textwrap.dedent("""\
        from typing import Annotated
        import torch
        from torchshapeflow import Shape
        Image = Annotated[torch.Tensor, Shape(3, 224, 224)]
    """)
    module = parse_source(src)
    raw = collect_raw_aliases(module)
    resolved = resolve_aliases(raw, {})
    assert "Image" in resolved
    assert str(resolved["Image"].shape) == "[3, 224, 224]"


def test_resolve_aliases_with_base_aliases() -> None:
    """Local alias can reference a pre-resolved imported alias."""

    src = textwrap.dedent("""\
        from typing import Annotated
        import torch
        from torchshapeflow import Shape
        # FeatureMap references nothing new; just tests that base aliases seed the table
        FeatureMap = Annotated[torch.Tensor, Shape("B", 64, "H", "W")]
    """)
    module = parse_source(src)
    raw = collect_raw_aliases(module)
    resolved = resolve_aliases(raw, {})
    assert "FeatureMap" in resolved
    assert str(resolved["FeatureMap"].shape) == "[B, 64, H, W]"


# ---------------------------------------------------------------------------
# extract_func_sig
# ---------------------------------------------------------------------------


def test_extract_func_sig_basic() -> None:
    src = textwrap.dedent("""\
        from typing import Annotated
        import torch
        from torchshapeflow import Shape

        Image = Annotated[torch.Tensor, Shape(3, 224, 224)]

        def process(x: Image) -> Annotated[torch.Tensor, Shape(64, 112, 112)]:
            pass
    """)
    module = parse_source(src)
    raw = collect_raw_aliases(module)
    aliases = resolve_aliases(raw, {})
    func_node = module.body[-1]
    import ast

    assert isinstance(func_node, ast.FunctionDef)
    sig = extract_func_sig(func_node, aliases)
    assert sig is not None
    assert sig.param_shapes[0] is not None
    assert str(sig.param_shapes[0].shape) == "[3, 224, 224]"
    assert sig.return_shape is not None
    assert str(sig.return_shape.shape) == "[64, 112, 112]"


def test_extract_func_sig_no_tensor_annotation_returns_none() -> None:
    src = textwrap.dedent("""\
        def add(a: int, b: int) -> int:
            return a + b
    """)
    module = parse_source(src)
    import ast

    func_node = module.body[0]
    assert isinstance(func_node, ast.FunctionDef)
    sig = extract_func_sig(func_node, {})
    assert sig is None


def test_extract_func_sig_mixed_params() -> None:
    """Non-tensor params produce None entries in param_shapes."""
    src = textwrap.dedent("""\
        from typing import Annotated
        import torch
        from torchshapeflow import Shape

        def forward(x: Annotated[torch.Tensor, Shape("B", 3, 32, 32)], scale: float):
            pass
    """)
    module = parse_source(src)
    import ast

    func_node = module.body[-1]
    assert isinstance(func_node, ast.FunctionDef)
    sig = extract_func_sig(func_node, {})
    assert sig is not None
    assert sig.param_shapes[0] is not None
    assert sig.param_shapes[1] is None


# ---------------------------------------------------------------------------
# ProjectIndex — same-file TypeAlias resolution (via analyze_source)
# ---------------------------------------------------------------------------


def test_same_file_typealias_used_as_param_annotation() -> None:
    """A TypeAlias defined in the same file seeds param shape inference."""
    src = textwrap.dedent("""\
        from typing import Annotated
        import torch
        from torchshapeflow import Shape

        Image = Annotated[torch.Tensor, Shape(3, 224, 224)]

        def crop(x: Image):
            y = x.permute(1, 2, 0)
    """)
    report = analyze_source(src, Path("mem.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[224, 224, 3]" for hover in report.hovers)


def test_same_file_typealias_annotated_assign_form() -> None:
    """`X: TypeAlias = ...` form also works."""
    src = textwrap.dedent("""\
        from typing import Annotated
        from typing_extensions import TypeAlias
        import torch
        from torchshapeflow import Shape

        Batch: TypeAlias = Annotated[torch.Tensor, Shape("B", 128)]

        def expand(x: Batch):
            y = x.unsqueeze(0)
    """)
    report = analyze_source(src, Path("mem.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[1, B, 128]" for hover in report.hovers)


def test_same_file_typealias_type_statement_form() -> None:
    """Python 3.12+ ``type X = ...`` aliases also seed param shape inference."""
    if not hasattr(ast, "TypeAlias"):
        pytest.skip("Python type statements require an AST runtime with ast.TypeAlias.")
    src = textwrap.dedent("""\
        from typing import Annotated
        import torch
        from torchshapeflow import Shape

        type Batch = Annotated[torch.Tensor, Shape("B", "T", 64)]

        def project(x: Batch):
            y = x.transpose(-2, -1)
    """)
    report = analyze_source(src, Path("mem.py"))
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[B, 64, T]" for hover in report.hovers)


# ---------------------------------------------------------------------------
# ProjectIndex — cross-file alias + function-signature inference
# ---------------------------------------------------------------------------


def test_cross_file_typealias(tmp_path: Path) -> None:
    """TypeAlias defined in one file resolves correctly when used in another."""
    types_py = tmp_path / "types.py"
    types_py.write_text(
        textwrap.dedent("""\
            from typing import Annotated
            import torch
            from torchshapeflow import Shape

            Image = Annotated[torch.Tensor, Shape(3, 224, 224)]
        """),
        encoding="utf-8",
    )

    main_py = tmp_path / "main.py"
    main_py.write_text(
        textwrap.dedent("""\
            from types_module import Image

            def process(x: Image):
                y = x.permute(1, 2, 0)
        """),
        encoding="utf-8",
    )

    # Patch the import name to resolve under tmp_path
    main_py.write_text(
        textwrap.dedent("""\
            from types import Image

            def process(x: Image):
                y = x.permute(1, 2, 0)
        """),
        encoding="utf-8",
    )
    # Use a real module name that matches the file
    (tmp_path / "types.py").write_text(
        textwrap.dedent("""\
            from typing import Annotated
            import torch
            from torchshapeflow import Shape
            Image = Annotated[torch.Tensor, Shape(3, 224, 224)]
        """),
        encoding="utf-8",
    )
    # Rename to avoid shadowing stdlib `types`
    shapes_py = tmp_path / "shapes.py"
    shapes_py.write_text(
        textwrap.dedent("""\
            from typing import Annotated
            import torch
            from torchshapeflow import Shape
            Image = Annotated[torch.Tensor, Shape(3, 224, 224)]
        """),
        encoding="utf-8",
    )

    consumer_py = tmp_path / "consumer.py"
    consumer_py.write_text(
        textwrap.dedent("""\
            from shapes import Image

            def process(x: Image):
                y = x.permute(1, 2, 0)
        """),
        encoding="utf-8",
    )

    project_index = ProjectIndex()
    from torchshapeflow.analyzer import analyze_path

    report = analyze_path(consumer_py, project_index)
    assert report.diagnostics == []
    assert any(hover.name == "y" and hover.shape == "[224, 224, 3]" for hover in report.hovers)


def test_cross_file_func_sig_inference(tmp_path: Path) -> None:
    """Calling an imported function uses its declared shapes for return inference."""
    helper_py = tmp_path / "helpers.py"
    helper_py.write_text(
        textwrap.dedent("""\
            from typing import Annotated
            import torch
            from torchshapeflow import Shape

            def embed(
                x: Annotated[torch.Tensor, Shape("B", "T")],
            ) -> Annotated[torch.Tensor, Shape("B", "T", 512)]:
                pass
        """),
        encoding="utf-8",
    )

    main_py = tmp_path / "main.py"
    main_py.write_text(
        textwrap.dedent("""\
            from typing import Annotated
            import torch
            from torchshapeflow import Shape
            from helpers import embed

            def run(tokens: Annotated[torch.Tensor, Shape("B", "T")]):
                out = embed(tokens)
        """),
        encoding="utf-8",
    )

    project_index = ProjectIndex()
    from torchshapeflow.analyzer import analyze_path

    report = analyze_path(main_py, project_index)
    assert report.diagnostics == []
    assert any(hover.name == "out" and hover.shape == "[B, T, 512]" for hover in report.hovers)


def test_cross_file_func_sig_constant_substitution(tmp_path: Path) -> None:
    """Concrete dim values are substituted into the return shape."""
    helper_py = tmp_path / "helpers.py"
    helper_py.write_text(
        textwrap.dedent("""\
            from typing import Annotated
            import torch
            from torchshapeflow import Shape

            def normalize(
                x: Annotated[torch.Tensor, Shape("B", "C", "H", "W")],
            ) -> Annotated[torch.Tensor, Shape("B", "C", "H", "W")]:
                pass
        """),
        encoding="utf-8",
    )

    main_py = tmp_path / "main.py"
    main_py.write_text(
        textwrap.dedent("""\
            from typing import Annotated
            import torch
            from torchshapeflow import Shape
            from helpers import normalize

            def run(img: Annotated[torch.Tensor, Shape(2, 3, 32, 32)]):
                out = normalize(img)
        """),
        encoding="utf-8",
    )

    project_index = ProjectIndex()
    from torchshapeflow.analyzer import analyze_path

    report = analyze_path(main_py, project_index)
    assert report.diagnostics == []
    assert any(hover.name == "out" and hover.shape == "[2, 3, 32, 32]" for hover in report.hovers)


def test_same_file_func_sig_inference() -> None:
    """Calling a local function with declared annotations propagates return shape."""
    src = textwrap.dedent("""\
        from typing import Annotated
        import torch
        from torchshapeflow import Shape

        def embed(
            x: Annotated[torch.Tensor, Shape("B", "T")],
        ) -> Annotated[torch.Tensor, Shape("B", "T", 512)]:
            pass

        def run(tokens: Annotated[torch.Tensor, Shape("B", "T")]):
            out = embed(tokens)
    """)
    report = analyze_source(src, Path("mem.py"))
    assert report.diagnostics == []
    assert any(hover.name == "out" and hover.shape == "[B, T, 512]" for hover in report.hovers)

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from torchshapeflow.model import Dim, SymbolicDim, TensorShape, TensorValue
from torchshapeflow.parser import AnnotationParseError, parse_source, parse_tensor_annotation


@dataclass(frozen=True)
class FuncSig:
    """Shape signature of a function derived from its type annotations.

    param_shapes: positional parameter shapes in declaration order; None for
        non-tensor or unannotated parameters.
    return_shape: declared return shape, or None if the return annotation is
        absent or not a tensor annotation.
    """

    param_shapes: tuple[TensorValue | None, ...]
    return_shape: TensorValue | None


@dataclass
class FileData:
    """Shape-relevant exports extracted from a single Python file."""

    aliases: dict[str, TensorValue] = field(default_factory=dict)
    func_sigs: dict[str, FuncSig] = field(default_factory=dict)


class ProjectIndex:
    """Lazily indexes Python source files on demand, caching by resolved path.

    Pass a single ProjectIndex instance to analyze_source / analyze_path so
    that cross-file alias and function-signature lookups share one cache for
    the lifetime of a check run.
    """

    def __init__(self) -> None:
        self._cache: dict[Path, FileData] = {}

    def index_file(self, path: Path) -> FileData:
        """Return the FileData for *path*, indexing it on first access."""
        resolved = path.resolve()
        if resolved not in self._cache:
            # Insert empty placeholder first to break import cycles.
            self._cache[resolved] = FileData()
            self._cache[resolved] = _index_file(resolved, self)
        return self._cache[resolved]

    def resolve_import(self, module_name: str, from_path: Path) -> Path | None:
        """Return the .py file that corresponds to *module_name*.

        Only searches relative to *from_path*'s directory — suitable for
        same-project imports. Returns None for third-party packages.
        """
        return _resolve_module_path(module_name, from_path)


_TYPE_ALIAS_NAMES: frozenset[str] = frozenset(
    {"TypeAlias", "typing.TypeAlias", "typing_extensions.TypeAlias"}
)


def build_file_data(
    module: ast.Module,
    path: Path,
    project_index: ProjectIndex | None = None,
) -> FileData:
    """Build alias and function-signature data for a parsed module.

    If *project_index* is provided, project-local ``from ... import ...``
    references are resolved first so imported aliases and annotated function
    signatures are available while processing the current module.
    """
    raw_aliases = collect_raw_aliases(module)
    import_map = collect_imports(module)

    imported_aliases: dict[str, TensorValue] = {}
    imported_funcs: dict[str, FuncSig] = {}
    if project_index is not None:
        for local_name, (module_name, original_name) in import_map.items():
            src_path = project_index.resolve_import(module_name, path)
            if src_path is not None:
                file_data = project_index.index_file(src_path)
                if original_name in file_data.aliases:
                    imported_aliases[local_name] = file_data.aliases[original_name]
                if original_name in file_data.func_sigs:
                    imported_funcs[local_name] = file_data.func_sigs[original_name]

    all_aliases = resolve_aliases(raw_aliases, imported_aliases)

    func_sigs: dict[str, FuncSig] = dict(imported_funcs)
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            sig = extract_func_sig(node, all_aliases)
            if sig is not None:
                func_sigs[node.name] = sig

    return FileData(aliases=all_aliases, func_sigs=func_sigs)


def unify_dims(declared: tuple[Dim, ...], actual: tuple[Dim, ...]) -> dict[str, Dim]:
    """Build a substitution mapping declared symbolic dims to actual dims.

    Only SymbolicDims in *declared* are mapped; constant and expression dims
    are skipped. Rank mismatches produce an empty mapping.

    Args:
        declared: Dim sequence from the function's declared parameter shape.
        actual: Dim sequence from the caller's argument shape.

    Returns:
        Substitution dict mapping symbolic name → actual Dim.
    """
    if len(declared) != len(actual):
        return {}
    return {d.name: a for d, a in zip(declared, actual, strict=True) if isinstance(d, SymbolicDim)}


def apply_substitution(shape: TensorShape, mapping: dict[str, Dim]) -> TensorShape:
    """Replace SymbolicDims in *shape* using *mapping*.

    Args:
        shape: Input shape potentially containing SymbolicDims.
        mapping: Substitution produced by unify_dims.

    Returns:
        New TensorShape with symbolic dims replaced where a mapping exists.
    """
    new_dims = tuple(
        mapping.get(dim.name, dim) if isinstance(dim, SymbolicDim) else dim for dim in shape.dims
    )
    return TensorShape(new_dims)


def extract_alias_binding(statement: ast.stmt) -> tuple[str, ast.AST] | None:
    """Return the alias name and RHS node for a single alias declaration."""
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            return target.id, statement.value
        return None
    if (
        isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.value is not None
        and _qualified_name(statement.annotation) in _TYPE_ALIAS_NAMES
    ):
        return statement.target.id, statement.value
    if hasattr(ast, "TypeAlias") and isinstance(statement, ast.TypeAlias):
        if isinstance(statement.name, ast.Name):
            return statement.name.id, statement.value
    return None


def collect_raw_aliases(module: ast.Module) -> dict[str, ast.AST]:
    """Collect module-level type alias assignments, returning name → RHS AST node.

    Handles both plain assignment (``X = Annotated[...]``) and annotated
    assignment (``X: TypeAlias = Annotated[...]``). On Python 3.12+ runtimes,
    ``type X = Annotated[...]`` statements are collected as well.
    """
    aliases: dict[str, ast.AST] = {}
    for node in module.body:
        alias = extract_alias_binding(node)
        if alias is not None:
            name, value = alias
            aliases[name] = value
    return aliases


def collect_imports(module: ast.Module) -> dict[str, tuple[str, str]]:
    """Return local_name → (module_name, original_name) for all from-imports."""
    imports: dict[str, tuple[str, str]] = {}
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                imports[local_name] = (node.module, alias.name)
    return imports


def resolve_aliases(
    raw_aliases: dict[str, ast.AST],
    base_aliases: dict[str, TensorValue],
) -> dict[str, TensorValue]:
    """Resolve raw alias AST nodes into TensorValues.

    *base_aliases* (e.g. already-resolved imported aliases) seeds the table
    so local aliases may reference them. Resolution proceeds in definition
    order; forward references within the same file are not supported.

    Args:
        raw_aliases: name → RHS AST node, from collect_raw_aliases.
        base_aliases: Pre-resolved aliases to seed the lookup table with.

    Returns:
        Combined dict of base_aliases plus any successfully resolved locals.
    """
    resolved: dict[str, TensorValue] = dict(base_aliases)
    for name, node in raw_aliases.items():
        try:
            tv = parse_tensor_annotation(node, resolved)
            if tv is not None:
                resolved[name] = tv
        except AnnotationParseError:
            pass
    return resolved


def extract_func_sig(func: ast.FunctionDef, aliases: dict[str, TensorValue]) -> FuncSig | None:
    """Extract the shape signature of a function from its type annotations.

    Returns None if the function has no tensor-annotated parameters or return.

    Args:
        func: The function definition AST node.
        aliases: Resolved alias table for resolving bare-name annotations.

    Returns:
        FuncSig with parameter and return shapes, or None if nothing is annotated.
    """
    param_shapes: list[TensorValue | None] = []
    has_any = False
    for arg in func.args.args:
        if arg.arg == "self":
            continue
        tv: TensorValue | None = None
        if arg.annotation is not None:
            try:
                tv = parse_tensor_annotation(arg.annotation, aliases)
            except AnnotationParseError:
                pass
        param_shapes.append(tv)
        if tv is not None:
            has_any = True
    return_shape: TensorValue | None = None
    if func.returns is not None:
        try:
            return_shape = parse_tensor_annotation(func.returns, aliases)
        except AnnotationParseError:
            pass
        if return_shape is not None:
            has_any = True
    if not has_any:
        return None
    return FuncSig(param_shapes=tuple(param_shapes), return_shape=return_shape)


def _index_file(path: Path, project_index: ProjectIndex) -> FileData:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return FileData()
    try:
        module = parse_source(source, str(path))
    except SyntaxError:
        return FileData()
    return build_file_data(module, path, project_index)


def _resolve_module_path(module_name: str, from_path: Path) -> Path | None:
    parts = module_name.split(".")
    base_dir = from_path.parent

    # "pkg.mod" → "pkg/mod.py"
    candidate = base_dir.joinpath(*parts[:-1], parts[-1] + ".py")
    if candidate.exists():
        return candidate

    # "pkg" → "pkg/__init__.py"
    package_init = base_dir.joinpath(*parts, "__init__.py")
    if package_init.exists():
        return package_init

    return None


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value)
        if not base:
            return node.attr
        return f"{base}.{node.attr}"
    return ""

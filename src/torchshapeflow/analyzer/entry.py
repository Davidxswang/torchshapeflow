"""Top-level orchestration for analyzing a Python source file."""

from __future__ import annotations

import ast
from pathlib import Path

from torchshapeflow.analysis_context import ModuleContext
from torchshapeflow.index import ProjectIndex, build_file_data, extract_func_sig
from torchshapeflow.parser import parse_source
from torchshapeflow.report import FileReport


def analyze_path(path: Path, project_index: ProjectIndex | None = None) -> FileReport:
    source = path.read_text(encoding="utf-8")
    return analyze_source(source, path, project_index)


def analyze_source(
    source: str,
    path: Path,
    project_index: ProjectIndex | None = None,
) -> FileReport:
    # Late imports to break the circular dependency: __init__.py re-exports
    # analyze_source from this module, while the walker layers it depends on
    # all import lower layers, never the other way round.
    from torchshapeflow.analyzer.functions import (
        analyze_function,
        emit_function_annotation_hovers,
    )
    from torchshapeflow.analyzer.modules import collect_class_specs, emit_module_alias_hovers

    module = parse_source(source, str(path))
    file_data = build_file_data(module, path, project_index)
    context = ModuleContext(path=path, aliases=file_data.aliases, func_sigs=file_data.func_sigs)
    emit_module_alias_hovers(module, context)
    class_specs, class_scalars, class_tensors = collect_class_specs(
        module, context, file_data.custom_module_templates
    )
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            analyze_function(node, context, {})
        elif isinstance(node, ast.ClassDef):
            specs = class_specs.get(node.name, {})
            context.self_scalars = class_scalars.get(node.name, {})
            context.self_tensors = class_tensors.get(node.name, {})
            # Pass 1: Collect method signatures for self.method() lookups.
            context.method_sigs = {}
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    sig = extract_func_sig(child, context.aliases)
                    if sig is not None:
                        context.method_sigs[child.name] = sig
            # Pass 2: Analyze all methods (except __init__ which is handled in specs).
            for child in node.body:
                if not isinstance(child, ast.FunctionDef):
                    continue
                if child.name == "__init__":
                    emit_function_annotation_hovers(child, context)
                    continue
                analyze_function(child, context, specs)
            context.method_sigs = {}
            context.self_scalars = {}
            context.self_tensors = {}
    return FileReport(
        path=str(path),
        diagnostics=context.diagnostics,
        hovers=context.hovers,
        suggestions=context.suggestions,
    )

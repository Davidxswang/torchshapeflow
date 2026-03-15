from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import typer

from torchshapeflow._version import __version__
from torchshapeflow.analyzer import analyze_path
from torchshapeflow.index import ProjectIndex
from torchshapeflow.report import FileReport
from torchshapeflow.utils.paths import collect_python_files

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False, no_args_is_help=True)


@app.command("check")
def check(
    path: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show per-file status for clean files."
    ),
) -> None:
    project_index = ProjectIndex()
    reports = [analyze_path(file_path, project_index) for file_path in collect_python_files(path)]
    payload = {"files": [report.to_dict() for report in reports]}
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(code=_exit_code(reports))
    lines: list[str] = []
    for report in reports:
        if not report.diagnostics:
            if verbose:
                lines.append(f"{report.path}: ok")
            continue
        for diagnostic in report.diagnostics:
            lines.append(
                f"{diagnostic.path}:{diagnostic.line}:{diagnostic.column} "
                f"{diagnostic.severity} {diagnostic.code} {diagnostic.message}"
            )
    lines.append(_summary(reports))
    typer.echo("\n".join(lines))
    raise typer.Exit(code=_exit_code(reports))


@app.command("version")
def version() -> None:
    typer.echo(__version__)


def _exit_code(reports: Sequence[FileReport]) -> int:
    for report in reports:
        if any(diagnostic.severity == "error" for diagnostic in report.diagnostics):
            return 1
    return 0


def _summary(reports: Sequence[FileReport]) -> str:
    """Build a human-readable summary line.

    Format examples:
        ``3 errors, 2 warnings in 2 files (15 files checked)``
        ``All clean (15 files checked)``
    """
    total_files = len(reports)
    errors = 0
    warnings = 0
    files_with_diagnostics = 0
    for report in reports:
        has_diag = False
        for diagnostic in report.diagnostics:
            if diagnostic.severity == "error":
                errors += 1
                has_diag = True
            elif diagnostic.severity == "warning":
                warnings += 1
                has_diag = True
        if has_diag:
            files_with_diagnostics += 1

    checked = f"({_plural(total_files, 'file')} checked)"

    if errors == 0 and warnings == 0:
        return f"All clean {checked}"

    parts: list[str] = []
    if errors:
        parts.append(_plural(errors, "error"))
    if warnings:
        parts.append(_plural(warnings, "warning"))
    return f"{', '.join(parts)} in {_plural(files_with_diagnostics, 'file')} {checked}"


def _plural(count: int, word: str) -> str:
    """Return e.g. ``1 error`` or ``3 errors``."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import typer

from torchshapeflow._version import __version__
from torchshapeflow.analyzer import analyze_path
from torchshapeflow.report import FileReport
from torchshapeflow.utils.paths import collect_python_files

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False, no_args_is_help=True)


@app.command("check")
def check(
    path: Path,
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    reports = [analyze_path(file_path) for file_path in collect_python_files(path)]
    payload = {"files": [report.to_dict() for report in reports]}
    if json_output:
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(code=_exit_code(reports))
    lines: list[str] = []
    for report in reports:
        if not report.diagnostics:
            lines.append(f"{report.path}: ok")
            continue
        for diagnostic in report.diagnostics:
            lines.append(
                f"{diagnostic.path}:{diagnostic.line}:{diagnostic.column} "
                f"{diagnostic.severity} {diagnostic.code} {diagnostic.message}"
            )
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

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from torchshapeflow._version import __version__
from torchshapeflow.analyzer import analyze_path
from torchshapeflow.index import ProjectIndex
from torchshapeflow.report import FileReport
from torchshapeflow.utils.paths import collect_python_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsf",
        description="Static AST-based PyTorch tensor shape analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser(
        "check",
        help="Run shape analysis on a file or directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    check_parser.add_argument("path", type=Path, help="File or directory to analyze.")
    check_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit JSON output.",
    )
    check_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-file status for clean files.",
    )

    subparsers.add_parser(
        "version",
        help="Print the installed TorchShapeFlow version.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = list(argv) if argv is not None else sys.argv[1:]
    if not args:
        parser.print_help()
        return 0

    namespace = parser.parse_args(args)
    command = namespace.command
    if command == "check":
        return _run_check(
            path=namespace.path,
            json_output=namespace.json_output,
            verbose=namespace.verbose,
        )
    if command == "version":
        print(__version__)
        return 0

    parser.print_help()
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


def _run_check(path: Path, json_output: bool, verbose: bool) -> int:
    project_index = ProjectIndex()
    reports = [analyze_path(file_path, project_index) for file_path in collect_python_files(path)]
    payload = {"files": [report.to_dict() for report in reports]}
    if json_output:
        print(json.dumps(payload, indent=2))
        return _exit_code(reports)

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
    print("\n".join(lines))
    return _exit_code(reports)


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

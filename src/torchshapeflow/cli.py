from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torchshapeflow._version import __version__
from torchshapeflow.analyzer import analyze_path
from torchshapeflow.index import ProjectIndex
from torchshapeflow.report import FileReport
from torchshapeflow.utils.paths import collect_python_files

_PY_FILE = re.compile(r"\.py$")


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

    suggest_parser = subparsers.add_parser(
        "suggest",
        help="Propose annotations TorchShapeFlow can already verify (JSON).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    suggest_parser.add_argument("path", type=Path, help="File or directory to analyze.")

    subparsers.add_parser(
        "mcp",
        help="Start an MCP server exposing TSF analysis to AI agents (requires the `mcp` extra).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers.add_parser(
        "version",
        help="Print the installed TorchShapeFlow version.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Internal subcommand used by the Claude Code plugin's PostToolUse hook.
    # ``help=argparse.SUPPRESS`` would be nicer but does not reliably hide
    # subparser entries across Python versions (3.10–3.12 render the literal
    # sentinel "==SUPPRESS==" instead of the documented hide behavior). A
    # plain descriptive help string is cross-version stable and makes the
    # underscore-prefixed name read correctly in ``tsf --help``.
    subparsers.add_parser(
        "_hook_post_edit",
        help="Claude Code plugin PostToolUse hook (internal; not for direct invocation).",
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
    if command == "suggest":
        return _run_suggest(path=namespace.path)
    if command == "mcp":
        return _run_mcp()
    if command == "_hook_post_edit":
        return _run_hook_post_edit()
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


def _run_suggest(path: Path) -> int:
    """Emit JSON proposals for annotations TorchShapeFlow can already verify.

    Suggestions are the analyzer's read-only proposals; TSF never writes them
    back to source.

    The payload includes ``diagnostics`` per file alongside ``suggestions`` so
    that a caller can tell an empty-but-clean analysis apart from an analysis
    that failed (e.g. a TSF1001 parse error on an unparseable annotation).
    The exit code mirrors ``tsf check``: non-zero when any file emits an
    ``error``-severity diagnostic. Without this, an agent calling
    ``tsf suggest`` on a broken file would see identical ``suggestions: []``
    and exit ``0`` as for a pristine file with nothing to add.
    """
    project_index = ProjectIndex()
    reports = [analyze_path(file_path, project_index) for file_path in collect_python_files(path)]
    payload = {
        "files": [
            {
                "path": report.path,
                "diagnostics": [item.to_dict() for item in report.diagnostics],
                "suggestions": [item.to_dict() for item in report.suggestions],
            }
            for report in reports
        ]
    }
    print(json.dumps(payload, indent=2))
    return _exit_code(reports)


def _run_mcp() -> int:
    """Start the MCP server on stdio.

    Lazy-imports ``mcp_server`` so callers that never invoke ``tsf mcp``
    don't need the optional ``mcp`` extra installed.
    """
    from torchshapeflow.mcp_server import run as run_mcp_server

    run_mcp_server()
    return 0


def _run_hook_post_edit() -> int:
    """Drive the Claude Code plugin's PostToolUse hook.

    Reads the Claude Code hook payload from stdin, extracts the edited file
    path, runs the analyzer on it, and prints the ``tsf check --json`` report
    to stderr **only when** an error-severity diagnostic is present. Always
    exits 0 so the hook never blocks the session on tooling hiccups (missing
    file, malformed payload, transient issues).

    The logic lives inside the CLI — rather than a standalone script invoked
    via ``python3`` — so the plugin's ``hooks/hooks.json`` can call
    ``uvx --from torchshapeflow[mcp] tsf _hook_post_edit``. One invocation,
    no separate Python interpreter on PATH, no shell expansion of
    ``${CLAUDE_PLUGIN_ROOT}``. Cross-platform everywhere ``uvx`` runs.
    """
    payload = _read_hook_payload()
    file_path = _extract_py_file_path(payload)
    if file_path is None:
        return 0
    target = Path(file_path)
    if not target.exists():
        return 0
    project_index = ProjectIndex()
    reports = [analyze_path(p, project_index) for p in collect_python_files(target)]
    if not any(_has_error(report) for report in reports):
        return 0
    report_payload = {"files": [report.to_dict() for report in reports]}
    # stderr so the diagnostic text flows through Claude Code's hook-output
    # channel without getting muddled with the hook's stdout contract.
    print(json.dumps(report_payload, indent=2), file=sys.stderr)
    return 0


def _read_hook_payload() -> dict[str, Any]:
    """Parse the Claude Code hook payload off stdin.

    Returns an empty dict on any malformed input so the hook degrades silently
    rather than spamming stderr during transient encoding issues.
    """
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _extract_py_file_path(payload: dict[str, Any]) -> str | None:
    """Extract ``tool_input.file_path`` from the hook payload iff it's a ``.py``.

    Claude Code's ``hooks.json`` ``if`` gate already filters on ``.py$``; this
    is a defence-in-depth check so the hook can't accidentally run on a
    non-Python file even if the gate is missed or changes shape.
    """
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    candidate = tool_input.get("file_path")
    if isinstance(candidate, str) and _PY_FILE.search(candidate):
        return candidate
    return None


def _has_error(report: FileReport) -> bool:
    return any(diag.severity == "error" for diag in report.diagnostics)


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

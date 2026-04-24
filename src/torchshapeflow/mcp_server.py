"""MCP server exposing TorchShapeFlow analysis to AI coding agents.

Install the optional dependency with ``pip install torchshapeflow[mcp]``, then
configure your agent runtime (Claude Code / Cursor / Aider / etc.) to launch
the server with ``tsf mcp``. The server speaks the standard MCP stdio
transport; no ports, no auth tokens.

The server exposes three tools:

- ``check(path)`` — run shape analysis; return diagnostics + hover facts.
- ``suggest(path)`` — return annotation proposals (same payload as
  ``tsf suggest``).
- ``hover_at(path, line, column)`` — return the inferred shape at a source
  location.

Each tool is a thin wrapper over ``analyze_path``; the analyzer's behavior
is identical to the corresponding CLI command. See
:doc:`../docs/agents` for the agent-facing workflow and
:doc:`../docs/architecture` for the diagnostic / suggestion JSON schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torchshapeflow.analyzer import analyze_path
from torchshapeflow.index import ProjectIndex
from torchshapeflow.utils.paths import collect_python_files

SERVER_NAME = "torchshapeflow"


def _analyze_target(path: str) -> list[Any]:
    """Run analysis over every ``.py`` file at *path* and return FileReports.

    Accepts either a single file path or a directory; mirrors the CLI's
    ``tsf check`` / ``tsf suggest`` semantics so MCP and CLI callers see the
    same result surface.
    """
    target = Path(path)
    project_index = ProjectIndex()
    return [analyze_path(p, project_index) for p in collect_python_files(target)]


def _tool_check(path: str) -> dict[str, Any]:
    """Build the ``check`` tool payload for *path*."""
    reports = _analyze_target(path)
    return {"files": [report.to_dict() for report in reports]}


def _tool_suggest(path: str) -> dict[str, Any]:
    """Build the ``suggest`` tool payload for *path*.

    Mirrors the ``tsf suggest`` CLI payload: each file entry carries both
    ``diagnostics`` and ``suggestions`` so an agent can distinguish
    "verified clean, nothing to add" from "analysis hit errors".
    """
    reports = _analyze_target(path)
    return {
        "files": [
            {
                "path": report.path,
                "diagnostics": [item.to_dict() for item in report.diagnostics],
                "suggestions": [item.to_dict() for item in report.suggestions],
            }
            for report in reports
        ]
    }


def _tool_hover_at(path: str, line: int, column: int) -> dict[str, Any] | None:
    """Return the hover fact that brackets *line*:*column* in *path*.

    Positions are 1-based (same convention as the analyzer's diagnostic
    output and editor columns). Returns ``None`` when no hover fact covers
    the requested location.
    """
    target = Path(path).resolve()
    for report in _analyze_target(path):
        if Path(report.path).resolve() != target:
            continue
        for hover in report.hovers:
            if hover.line <= line <= hover.end_line and hover.column <= column <= hover.end_column:
                return hover.to_dict()
    return None


def build_server() -> Any:
    """Build and return the FastMCP server instance.

    Imported lazily so ``torchshapeflow`` can be used without installing the
    optional ``mcp`` extra.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without the extra.
        raise RuntimeError(
            "The `mcp` package is required to run the TSF MCP server. "
            "Install it with `pip install torchshapeflow[mcp]`."
        ) from exc

    server = FastMCP(SERVER_NAME)

    @server.tool()
    def check(path: str) -> dict[str, Any]:
        """Run shape analysis on the file or directory at *path*.

        Returns per-file diagnostics (with structured ``expected`` /
        ``actual`` / ``hint`` fields on shape mismatches) and inferred-shape
        hovers. Use this when a user has just edited PyTorch code or asks
        whether a function's shapes are consistent.
        """
        return _tool_check(path)

    @server.tool()
    def suggest(path: str) -> dict[str, Any]:
        """Propose return annotations TorchShapeFlow has already verified.

        Each suggestion includes a pasteable ``annotation`` string rendered
        using the target file's existing import spelling. TSF never writes
        source — the agent decides whether to apply. Inspect ``diagnostics``
        alongside ``suggestions``: an empty ``suggestions`` list alongside
        error diagnostics means TSF could not verify the function.
        """
        return _tool_suggest(path)

    @server.tool()
    def hover_at(path: str, line: int, column: int) -> dict[str, Any] | None:
        """Return the hover fact (inferred tensor shape) at *path*:*line*:*column*.

        Positions are 1-based. Use this to look up the inferred shape at a
        specific source location (e.g., the variable the user is asking
        about). Returns ``None`` when no hover covers the requested position.
        """
        return _tool_hover_at(path, line, column)

    return server


def run() -> None:
    """Start the MCP server on stdio.

    Intended for use from ``tsf mcp``. Blocks until the client closes
    stdin.
    """
    build_server().run()

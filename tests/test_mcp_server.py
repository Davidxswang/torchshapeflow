"""Tests for the MCP tool helpers.

The MCP transport itself is the SDK's responsibility; we only test the
payload-building wrappers around ``analyze_path``. If these behave like the
CLI counterparts, the MCP server's three tools behave correctly by
construction.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

mcp_server = pytest.importorskip("torchshapeflow.mcp_server")


_CLEAN_SOURCE = dedent(
    """
    from typing import Annotated
    import torch
    from torchshapeflow import Shape


    def fn(x: Annotated[torch.Tensor, Shape("B", "T", 768)]):
        return x
    """
).lstrip()


_BROKEN_SOURCE = dedent(
    """
    from typing import Annotated
    import torch
    import torch.nn as nn
    from torchshapeflow import Shape


    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(768, 256)

        def forward(self, x: Annotated[torch.Tensor, Shape("B", "T", 512)]):
            return self.fc(x)
    """
).lstrip()


def _write(tmp_path: Path, name: str, source: str) -> Path:
    target = tmp_path / name
    target.write_text(source, encoding="utf-8")
    return target


def test_check_tool_returns_diagnostics_and_hovers(tmp_path: Path) -> None:
    """check() mirrors tsf check --json: diagnostics and hovers per file."""
    target = _write(tmp_path, "clean.py", _CLEAN_SOURCE)
    payload = mcp_server._tool_check(str(target))
    assert isinstance(payload, dict)
    assert len(payload["files"]) == 1
    file_payload = payload["files"][0]
    assert file_payload["diagnostics"] == []
    # The annotated parameter produces at least a signature hover.
    assert any(h["kind"] == "signature" for h in file_payload["hovers"])


def test_check_tool_surfaces_errors(tmp_path: Path) -> None:
    """A shape mismatch shows up in the check payload with structured fields."""
    target = _write(tmp_path, "broken.py", _BROKEN_SOURCE)
    payload = mcp_server._tool_check(str(target))
    diagnostics = payload["files"][0]["diagnostics"]
    assert diagnostics
    tsf1007 = next(d for d in diagnostics if d["code"] == "TSF1007")
    assert tsf1007["severity"] == "error"
    assert tsf1007["expected"] == "last dim = 768"
    assert "512" in tsf1007["actual"]
    assert "hint" in tsf1007


def test_suggest_tool_returns_diagnostics_and_suggestions(tmp_path: Path) -> None:
    """suggest() mirrors tsf suggest: diagnostics + suggestions per file."""
    target = _write(tmp_path, "clean.py", _CLEAN_SOURCE)
    payload = mcp_server._tool_suggest(str(target))
    file_payload = payload["files"][0]
    assert file_payload["diagnostics"] == []
    suggestions = file_payload["suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["function"] == "fn"
    assert suggestions[0]["annotation"] == ("Annotated[torch.Tensor, Shape('B', 'T', 768)]")


def test_suggest_tool_stays_silent_on_broken_function(tmp_path: Path) -> None:
    """A function that TSF flagged must not receive a suggestion."""
    target = _write(tmp_path, "broken.py", _BROKEN_SOURCE)
    payload = mcp_server._tool_suggest(str(target))
    assert payload["files"][0]["diagnostics"]
    assert payload["files"][0]["suggestions"] == []


def test_hover_at_tool_returns_fact_when_cursor_on_parameter(tmp_path: Path) -> None:
    """hover_at() over an annotated parameter returns its inferred shape."""
    target = _write(tmp_path, "clean.py", _CLEAN_SOURCE)
    # Line 6 is `def fn(x: Annotated[...])`. Column of `x` is 8 (1-based).
    payload = mcp_server._tool_hover_at(str(target), 6, 8)
    assert payload is not None
    assert payload["name"] == "x"
    assert payload["shape"] == "[B, T, 768]"


def test_hover_at_tool_returns_none_outside_any_hover(tmp_path: Path) -> None:
    """Positions outside any hover range return None."""
    target = _write(tmp_path, "clean.py", _CLEAN_SOURCE)
    # Line 1 is `from typing import Annotated` — no hover there.
    assert mcp_server._tool_hover_at(str(target), 1, 1) is None


def test_build_server_wires_three_tools() -> None:
    """The FastMCP server exposes exactly the three documented tools."""
    server = mcp_server.build_server()
    # FastMCP's public tool registry lives on the server's underlying manager.
    # We exercise it through the public list_tools coroutine for stability.
    import asyncio

    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    assert names == {"check", "suggest", "hover_at"}


def test_check_tool_graceful_on_missing_path(tmp_path: Path) -> None:
    """Agents pass bad paths; crashing the server would kill the session."""
    missing = tmp_path / "does-not-exist.py"
    payload = mcp_server._tool_check(str(missing))
    assert payload == {"files": []}


def test_suggest_tool_graceful_on_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.py"
    payload = mcp_server._tool_suggest(str(missing))
    assert payload == {"files": []}


def test_hover_at_tool_graceful_on_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.py"
    assert mcp_server._tool_hover_at(str(missing), 1, 1) is None

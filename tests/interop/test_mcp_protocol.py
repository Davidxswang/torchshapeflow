"""MCP stdio protocol integration test.

Spawns ``tsf mcp`` as a subprocess, completes the standard MCP handshake
(``initialize`` → ``notifications/initialized`` → ``tools/list``), then
calls one tool (``check``) over JSON-RPC and verifies the response shape.

Closes the gap left in PR #31's unit tests, which exercised the
``_tool_*`` helpers directly but not the actual stdio wire protocol.
Catches breakage if the FastMCP SDK changes its envelope, response format,
or initialization semantics.

Skips when the optional ``mcp`` extra isn't installed in the test
environment.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .conftest import fixture_path, require_binary

mcp = pytest.importorskip("mcp")  # noqa: F841 — only used to gate the test


_FIXTURE = "bug_linear_in_features.py"


def _send_jsonrpc(proc: subprocess.Popen[str], message: dict[str, Any]) -> None:
    """Write a JSON-RPC frame on the server's stdin (newline-delimited)."""
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _read_response(
    proc: subprocess.Popen[str], expected_id: int, *, timeout_s: float = 10.0
) -> dict[str, Any]:
    """Read newline-delimited JSON-RPC frames until one matches ``expected_id``.

    The server may emit notifications between requests; skip anything whose
    id doesn't match. A simple line-budget guard keeps the test from hanging
    on a misbehaving server in CI.

    Returns ``dict[str, Any]`` — JSON-RPC payloads from an external server are
    genuinely dynamic, so ``Any`` is the legitimate choice (matches the
    same idiom used in ``src/torchshapeflow/cli.py`` for the hook payload).
    """
    assert proc.stdout is not None
    for _ in range(64):
        line = proc.stdout.readline()
        if not line:
            raise AssertionError("MCP server closed stdout before responding")
        message = json.loads(line)
        if message.get("id") == expected_id:
            return message
    raise AssertionError(f"no response with id={expected_id} after 64 frames")


def test_mcp_stdio_initialize_list_tools_call(tmp_path: Path) -> None:
    """End-to-end MCP handshake + a tools/call round trip."""
    tsf = require_binary("tsf")
    proc = subprocess.Popen(
        [tsf, "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered so readline returns frames promptly
    )
    try:
        # 1. initialize
        _send_jsonrpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "interop-test", "version": "0"},
                },
            },
        )
        init_response = _read_response(proc, expected_id=1)
        result: Any = init_response["result"]
        assert result["serverInfo"]["name"] == "torchshapeflow"
        assert "tools" in result["capabilities"]

        # 2. handshake completion notification (no response expected)
        _send_jsonrpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 3. tools/list — our three tools must be registered.
        _send_jsonrpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        list_response = _read_response(proc, expected_id=2)
        list_result: Any = list_response["result"]
        names = {tool["name"] for tool in list_result["tools"]}
        assert names == {"check", "suggest", "hover_at"}

        # 4. tools/call check — should return the structured TSF1007.
        _send_jsonrpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "check",
                    "arguments": {"path": str(fixture_path(_FIXTURE))},
                },
            },
        )
        call_response = _read_response(proc, expected_id=3)
        call_result: Any = call_response["result"]
        text_block = call_result["content"][0]
        # FastMCP wraps dict tool results in a TextContent with a JSON string.
        assert text_block["type"] == "text"
        check_payload = json.loads(text_block["text"])
        diagnostics = check_payload["files"][0]["diagnostics"]
        assert any(diag["code"] == "TSF1007" for diag in diagnostics)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

#!/usr/bin/env python3
"""PostToolUse hook: surface shape diagnostics after the agent edits a Python file.

Invoked by Claude Code after ``Write`` / ``Edit`` tool uses on ``.py`` files
(see ``hooks/hooks.json``). Reads the hook payload on stdin, runs
``tsf check --json`` on the touched file via ``uvx``, and — only when the
analyzer surfaced error-severity diagnostics — prints the JSON report to
stderr so the conversation sees it without polluting clean edits.

Design notes:

- Hook exits 0 unconditionally: hooks that block or error out on tooling
  problems (missing ``uvx``, network hiccups) would make the plugin feel
  fragile.  Silent-on-missing is the safe default; the MCP tools remain
  available for explicit invocation.
- Only errors are printed; warnings (``TSF2001`` et al.) are informational
  and handled by the skill on demand.
- Runs under a short timeout to avoid stalling the session on pathological
  inputs. 30s is ample for a single-file ``tsf check``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_PY_FILE = re.compile(r"\.py$")
_TIMEOUT_SECONDS = 30


def _read_payload() -> dict[str, Any]:
    # The hook payload is defined by Claude Code's runtime, not by this
    # plugin — treat it as unstructured JSON and validate each field we
    # read. That is the legitimate use case for Any here.
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _extract_path(payload: dict[str, Any]) -> str | None:
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    candidate = tool_input.get("file_path")
    if isinstance(candidate, str) and _PY_FILE.search(candidate):
        return candidate
    return None


def _run_tsf_check(file_path: str) -> str | None:
    """Run ``tsf check --json`` on *file_path*. Return stdout, or None on any
    failure that should be treated as silent (missing uvx, timeout, etc.)."""
    try:
        result = subprocess.run(
            [
                "uvx",
                "--from",
                "torchshapeflow[mcp]",
                "tsf",
                "check",
                "--json",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    return result.stdout


def _has_error_diagnostic(stdout: str) -> bool:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return False
    files = payload.get("files")
    if not isinstance(files, list):
        return False
    for file_payload in files:
        if not isinstance(file_payload, dict):
            continue
        diagnostics = file_payload.get("diagnostics")
        if not isinstance(diagnostics, list):
            continue
        for diag in diagnostics:
            if isinstance(diag, dict) and diag.get("severity") == "error":
                return True
    return False


def main() -> int:
    payload = _read_payload()
    file_path = _extract_path(payload)
    if not file_path:
        return 0
    if not Path(file_path).exists():
        return 0
    stdout = _run_tsf_check(file_path)
    if stdout is None or not stdout.strip():
        return 0
    if _has_error_diagnostic(stdout):
        # Surface the structured report on stderr so the conversation sees
        # the diagnostic fields (expected/actual/hint) without being forced
        # to re-read the file.
        print(stdout, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

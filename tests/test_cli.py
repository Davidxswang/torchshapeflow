from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from torchshapeflow.cli import main


def _run_cli(*args: str) -> tuple[int, str]:
    stdout = StringIO()
    with redirect_stdout(stdout):
        exit_code = main(args)
    return exit_code, stdout.getvalue()


def test_cli_json_output() -> None:
    exit_code, stdout = _run_cli("check", str(Path("examples/error_cases.py")), "--json")
    assert exit_code == 1
    payload = json.loads(stdout)
    assert "files" in payload
    assert payload["files"][0]["diagnostics"]


def test_cli_default_no_ok_line() -> None:
    """Default mode suppresses 'ok' lines for clean files."""
    exit_code, stdout = _run_cli("check", str(Path("examples/simple_cnn.py")))
    assert exit_code == 0
    assert "ok" not in stdout
    assert "All clean (1 file checked)" in stdout


def test_cli_verbose_shows_ok() -> None:
    """--verbose shows 'ok' lines for clean files plus summary."""
    exit_code, stdout = _run_cli("check", str(Path("examples/simple_cnn.py")), "--verbose")
    assert exit_code == 0
    assert "simple_cnn.py: ok" in stdout
    assert "All clean (1 file checked)" in stdout


def test_cli_verbose_short_flag() -> None:
    """-v is a short alias for --verbose."""
    exit_code, stdout = _run_cli("check", str(Path("examples/simple_cnn.py")), "-v")
    assert exit_code == 0
    assert "simple_cnn.py: ok" in stdout


def test_cli_summary_with_errors() -> None:
    """Summary line reports error and warning counts."""
    exit_code, stdout = _run_cli("check", str(Path("examples/error_cases.py")))
    assert exit_code == 1
    assert "TSF" in stdout
    assert "in 1 file" in stdout
    assert "checked" in stdout


def test_cli_json_no_summary() -> None:
    """JSON mode does not include a summary line."""
    exit_code, stdout = _run_cli("check", str(Path("examples/simple_cnn.py")), "--json")
    assert exit_code == 0
    payload = json.loads(stdout)
    assert "files" in payload
    assert "All clean" not in stdout


def test_cli_version() -> None:
    exit_code, stdout = _run_cli("version")
    assert exit_code == 0
    assert stdout.strip()

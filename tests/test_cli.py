from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from torchshapeflow.cli import app


def test_cli_json_output() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["check", str(Path("examples/error_cases.py")), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert "files" in payload
    assert payload["files"][0]["diagnostics"]


def test_cli_default_no_ok_line() -> None:
    """Default mode suppresses 'ok' lines for clean files."""
    runner = CliRunner()
    result = runner.invoke(app, ["check", str(Path("examples/simple_cnn.py"))])
    assert result.exit_code == 0
    assert "ok" not in result.stdout
    assert "All clean (1 file checked)" in result.stdout


def test_cli_verbose_shows_ok() -> None:
    """--verbose shows 'ok' lines for clean files plus summary."""
    runner = CliRunner()
    result = runner.invoke(app, ["check", str(Path("examples/simple_cnn.py")), "--verbose"])
    assert result.exit_code == 0
    assert "simple_cnn.py: ok" in result.stdout
    assert "All clean (1 file checked)" in result.stdout


def test_cli_verbose_short_flag() -> None:
    """-v is a short alias for --verbose."""
    runner = CliRunner()
    result = runner.invoke(app, ["check", str(Path("examples/simple_cnn.py")), "-v"])
    assert result.exit_code == 0
    assert "simple_cnn.py: ok" in result.stdout


def test_cli_summary_with_errors() -> None:
    """Summary line reports error and warning counts."""
    runner = CliRunner()
    result = runner.invoke(app, ["check", str(Path("examples/error_cases.py"))])
    assert result.exit_code == 1
    # Should contain diagnostic lines
    assert "TSF" in result.stdout
    # Should contain summary with counts and "in 1 file"
    assert "in 1 file" in result.stdout
    assert "checked" in result.stdout


def test_cli_json_no_summary() -> None:
    """JSON mode does not include a summary line."""
    runner = CliRunner()
    result = runner.invoke(app, ["check", str(Path("examples/simple_cnn.py")), "--json"])
    assert result.exit_code == 0
    # Should be valid JSON (no extra text)
    payload = json.loads(result.stdout)
    assert "files" in payload
    assert "All clean" not in result.stdout

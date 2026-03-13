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

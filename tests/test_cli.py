from __future__ import annotations

import json
import sys
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


def test_cli_suggest_emits_json(tmp_path: Path) -> None:
    """`tsf suggest` emits proposals as JSON without touching source."""
    source = (
        "from typing import Annotated\n"
        "import torch\n"
        "from torchshapeflow import Shape\n"
        "\n"
        "def fn(x: Annotated[torch.Tensor, Shape('B', 'T', 768)]):\n"
        "    return x\n"
    )
    target = tmp_path / "m.py"
    target.write_text(source, encoding="utf-8")
    exit_code, stdout = _run_cli("suggest", str(target))
    assert exit_code == 0
    payload = json.loads(stdout)
    assert len(payload["files"]) == 1
    file_payload = payload["files"][0]
    assert file_payload["diagnostics"] == []
    suggestions = file_payload["suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["function"] == "fn"
    assert suggestions[0]["annotation"] == ("Annotated[torch.Tensor, Shape('B', 'T', 768)]")
    # The command is read-only: source on disk is unchanged.
    assert target.read_text(encoding="utf-8") == source


def test_cli_suggest_clean_file_is_success(tmp_path: Path) -> None:
    """A file with no opportunities returns an empty suggestions list and exit 0."""
    source = (
        "from typing import Annotated\n"
        "import torch\n"
        "from torchshapeflow import Shape\n"
        "\n"
        "def fn(\n"
        "    x: Annotated[torch.Tensor, Shape('B',)],\n"
        ") -> Annotated[torch.Tensor, Shape('B',)]:\n"
        "    return x\n"
    )
    target = tmp_path / "m.py"
    target.write_text(source, encoding="utf-8")
    exit_code, stdout = _run_cli("suggest", str(target))
    assert exit_code == 0
    payload = json.loads(stdout)
    file_payload = payload["files"][0]
    assert file_payload["suggestions"] == []
    assert file_payload["diagnostics"] == []


def test_cli_suggest_surfaces_errors_and_nonzero_exit(tmp_path: Path) -> None:
    """A file with a TSF-error diagnostic exposes it in JSON and exits non-zero.

    Ensures an agent calling `tsf suggest` can distinguish "analysis succeeded
    and there was nothing to suggest" from "analysis found a problem" — a
    silent success would mask real shape bugs.
    """
    source = (
        "from typing import Annotated\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "from torchshapeflow import Shape\n"
        "\n"
        "class M(nn.Module):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(768, 256)\n"
        "\n"
        "    def forward(self, x: Annotated[torch.Tensor, Shape('B', 'T', 512)]):\n"
        "        return self.fc(x)\n"
    )
    target = tmp_path / "m.py"
    target.write_text(source, encoding="utf-8")
    exit_code, stdout = _run_cli("suggest", str(target))
    assert exit_code == 1
    payload = json.loads(stdout)
    diagnostics = payload["files"][0]["diagnostics"]
    assert diagnostics
    assert any(d["code"] == "TSF1007" for d in diagnostics)


def test_cli_check_json_does_not_include_suggestions(tmp_path: Path) -> None:
    """The shared report JSON (tsf check --json) stays narrow — no suggestions."""
    source = (
        "from typing import Annotated\n"
        "import torch\n"
        "from torchshapeflow import Shape\n"
        "\n"
        "def fn(x: Annotated[torch.Tensor, Shape('B',)]):\n"
        "    return x\n"
    )
    target = tmp_path / "m.py"
    target.write_text(source, encoding="utf-8")
    exit_code, stdout = _run_cli("check", str(target), "--json")
    assert exit_code == 0
    payload = json.loads(stdout)
    assert "suggestions" not in payload["files"][0]


def _run_cli_with_stdin(stdin_payload: str, *args: str) -> tuple[int, str, str]:
    """Invoke ``tsf`` with a custom stdin payload. Returns (exit, stdout, stderr)."""
    from contextlib import redirect_stderr

    from torchshapeflow.cli import main

    original_stdin = sys.stdin
    stdout = StringIO()
    stderr = StringIO()
    try:
        sys.stdin = StringIO(stdin_payload)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = main(args)
    finally:
        sys.stdin = original_stdin
    return exit_code, stdout.getvalue(), stderr.getvalue()


def test_hook_post_edit_silent_on_clean_file(tmp_path: Path) -> None:
    """No error diagnostics → hook stays quiet and exits 0."""
    source = (
        "from typing import Annotated\n"
        "import torch\n"
        "from torchshapeflow import Shape\n"
        "\n"
        "def fn(x: Annotated[torch.Tensor, Shape('B',)]):\n"
        "    return x\n"
    )
    target = tmp_path / "clean.py"
    target.write_text(source, encoding="utf-8")
    payload = json.dumps({"tool_input": {"file_path": str(target)}})
    exit_code, stdout, stderr = _run_cli_with_stdin(payload, "_hook_post_edit")
    assert exit_code == 0
    assert stdout == ""
    assert stderr == ""


def test_hook_post_edit_emits_diagnostics_on_shape_error(tmp_path: Path) -> None:
    """Error-severity diagnostics → hook prints the JSON report to stderr, exit 0."""
    source = (
        "from typing import Annotated\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "from torchshapeflow import Shape\n"
        "\n"
        "class M(nn.Module):\n"
        "    def __init__(self) -> None:\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(768, 256)\n"
        "\n"
        "    def forward(self, x: Annotated[torch.Tensor, Shape('B', 'T', 512)]):\n"
        "        return self.fc(x)\n"
    )
    target = tmp_path / "broken.py"
    target.write_text(source, encoding="utf-8")
    payload = json.dumps({"tool_input": {"file_path": str(target)}})
    exit_code, stdout, stderr = _run_cli_with_stdin(payload, "_hook_post_edit")
    # Hook never blocks the session — exit is always 0.
    assert exit_code == 0
    # Report goes to stderr so it doesn't muddle the hook's stdout channel.
    assert stdout == ""
    assert stderr, "expected the hook to surface the diagnostic report on stderr"
    report = json.loads(stderr)
    assert any(
        diag["code"] == "TSF1007"
        for file_payload in report["files"]
        for diag in file_payload["diagnostics"]
    )


def test_hook_post_edit_ignores_non_python_file(tmp_path: Path) -> None:
    """Defence in depth: the hook won't act on a non-.py path even if invoked."""
    target = tmp_path / "notes.md"
    target.write_text("# not python\n", encoding="utf-8")
    payload = json.dumps({"tool_input": {"file_path": str(target)}})
    exit_code, stdout, stderr = _run_cli_with_stdin(payload, "_hook_post_edit")
    assert exit_code == 0
    assert stdout == ""
    assert stderr == ""


def test_hook_post_edit_ignores_missing_file(tmp_path: Path) -> None:
    """A payload pointing at a missing file is a no-op, not a crash."""
    payload = json.dumps({"tool_input": {"file_path": str(tmp_path / "gone.py")}})
    exit_code, stdout, stderr = _run_cli_with_stdin(payload, "_hook_post_edit")
    assert exit_code == 0
    assert stdout == ""
    assert stderr == ""


def test_hook_post_edit_handles_malformed_payload() -> None:
    """Bad JSON on stdin is silently ignored; hook returns 0."""
    exit_code, stdout, stderr = _run_cli_with_stdin("not json at all", "_hook_post_edit")
    assert exit_code == 0
    assert stdout == ""
    assert stderr == ""

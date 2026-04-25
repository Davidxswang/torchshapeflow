"""Type-checker interop tests: pyright, mypy, ty against TSF-annotated code.

The contract: TSF's ``Annotated[torch.Tensor, Shape(...)]`` must be a no-op
to other type-checkers — the base type is still ``torch.Tensor``, and the
``Shape(...)`` metadata is part of PEP 593, which all three checkers accept.
This guards against regressions where a parser or example change accidentally
breaks the Cursor / VS Code (Pylance/pyright) experience for downstream
users.

Each test invokes the checker as a subprocess against fixtures from
``tests/interop/fixtures/`` and asserts:

- Clean fixtures: zero errors reported.
- Bug fixtures: still zero errors. Shape mismatches are TSF's domain;
  other checkers don't see them, and we want to make sure we don't make
  them produce false positives either.

Skips when:
- ``torch`` isn't importable (the fixtures import torch; checkers can't
  resolve it without the package installed).
- The checker binary isn't on PATH (lets local dev without all three
  installed run the rest of the suite).
"""

from __future__ import annotations

import json
import subprocess

import pytest

from .conftest import (
    ALL_FIXTURES,
    fixture_path,
    require_binary,
)

# The fixtures import torch; without it installed, pyright/mypy/ty can't
# resolve ``torch.Tensor`` and produce false-positive import errors that
# have nothing to do with our annotations.
pytest.importorskip("torch")


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_pyright_reports_zero_errors(fixture: str) -> None:
    """pyright via the PyPI wrapper. Uses ``--outputjson`` so we don't have
    to scrape unstructured text."""
    pyright = require_binary("pyright")
    result = subprocess.run(
        [pyright, "--outputjson", str(fixture_path(fixture))],
        capture_output=True,
        text=True,
        check=False,
    )
    # pyright exits non-zero when it finds errors, but its JSON output is
    # the source of truth — parse stdout regardless of exit code.
    payload = json.loads(result.stdout)
    error_count = payload["summary"]["errorCount"]
    assert error_count == 0, (
        f"pyright reported {error_count} errors on {fixture}: "
        f"{payload.get('generalDiagnostics', [])}"
    )


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_mypy_reports_zero_errors(fixture: str) -> None:
    """mypy with default config — the most common production type-checker."""
    mypy = require_binary("mypy")
    result = subprocess.run(
        [
            mypy,
            "--no-incremental",  # avoids cache pollution between fixture runs
            "--show-error-codes",
            str(fixture_path(fixture)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # mypy exits 0 on success, 1 on errors found, >=2 on internal failures.
    # Treat 0 (success) and 2+ (mypy itself broken) differently from 1
    # (real type errors found).
    assert result.returncode == 0, (
        f"mypy reported errors on {fixture}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.parametrize("fixture", ALL_FIXTURES)
def test_ty_reports_zero_errors(fixture: str) -> None:
    """ty (Astral's type checker) — already a TSF dev dependency."""
    ty = require_binary("ty")
    result = subprocess.run(
        [ty, "check", str(fixture_path(fixture))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"ty reported errors on {fixture}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

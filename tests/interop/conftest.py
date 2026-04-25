"""Shared test helpers for ``tests/interop/``.

Pinned constants (fixture paths, which fixtures are clean vs deliberately
broken) and small helpers for skipping when external binaries / packages
aren't available locally.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

CLEAN_FIXTURES: list[str] = [
    "clean_transformer.py",
    "clean_cnn.py",
    "clean_typealias.py",
]

BUG_FIXTURES: list[str] = [
    "bug_linear_in_features.py",
    "bug_matmul_inner.py",
]

ALL_FIXTURES: list[str] = CLEAN_FIXTURES + BUG_FIXTURES


def fixture_path(name: str) -> Path:
    return FIXTURES_DIR / name


def require_binary(name: str) -> str:
    """Return the binary path, or skip the test when not on PATH.

    Lets developers without pyright / mypy installed locally still run the
    rest of the suite. CI installs them so the assertions actually run there.
    """
    found = shutil.which(name)
    if found is None:
        pytest.skip(f"`{name}` not on PATH; install it to run this test")
    return found

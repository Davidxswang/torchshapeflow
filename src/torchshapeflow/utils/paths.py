from __future__ import annotations

from pathlib import Path


def collect_python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.rglob("*.py") if candidate.is_file())

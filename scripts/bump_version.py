from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import cast

PYPROJECT_PATH = Path("pyproject.toml")
PYTHON_VERSION_PATH = Path("src/torchshapeflow/_version.py")
EXTENSION_PACKAGE_PATH = Path("extensions/vscode/package.json")
EXTENSION_LOCK_PATH = Path("extensions/vscode/package-lock.json")
VERSION_PATTERN = re.compile(r'^version = "(?P<version>\d+\.\d+\.\d+)"$', re.MULTILINE)
PYTHON_VERSION_PATTERN = re.compile(
    r'^__version__ = "(?P<version>\d+)\.(\d+)\.(\d+)"$',
    re.MULTILINE,
)


def bump(kind: str, version: tuple[int, int, int]) -> tuple[int, int, int]:
    major, minor, patch = version
    if kind == "patch":
        return major, minor, patch + 1
    if kind == "minor":
        return major, minor + 1, 0
    if kind == "major":
        return major + 1, 0, 0
    raise ValueError(f"Unsupported bump kind: {kind}")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: bump_version.py [patch|minor|major]")

    kind = sys.argv[1]
    content = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(content)
    if match is None:
        raise SystemExit("Could not locate version in pyproject.toml")

    version_parts = tuple(int(part) for part in match.group("version").split("."))
    if len(version_parts) != 3:
        raise SystemExit("Version must have three components.")
    version = (version_parts[0], version_parts[1], version_parts[2])
    next_version = bump(kind, version)
    version_text = f"{next_version[0]}.{next_version[1]}.{next_version[2]}"
    replacement = f'version = "{version_text}"'
    updated = VERSION_PATTERN.sub(replacement, content, count=1)
    PYPROJECT_PATH.write_text(updated, encoding="utf-8")
    _update_python_version(version_text)
    _update_extension_version(version_text)
    print(replacement)
    return 0


def _update_python_version(version_text: str) -> None:
    content = PYTHON_VERSION_PATH.read_text(encoding="utf-8")
    updated = PYTHON_VERSION_PATTERN.sub(f'__version__ = "{version_text}"', content, count=1)
    PYTHON_VERSION_PATH.write_text(updated, encoding="utf-8")


def _update_extension_version(version_text: str) -> None:
    for path in (EXTENSION_PACKAGE_PATH, EXTENSION_LOCK_PATH):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        _set_version_fields(payload, version_text)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _set_version_fields(payload: dict[str, object], version_text: str) -> None:
    payload["version"] = version_text
    packages = payload.get("packages")
    if isinstance(packages, dict):
        package_map = cast(dict[str, object], packages)
        root = package_map.get("")
        if isinstance(root, dict):
            root_payload = cast(dict[str, object], root)
            root_payload["version"] = version_text


if __name__ == "__main__":
    raise SystemExit(main())

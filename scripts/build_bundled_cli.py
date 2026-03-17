from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetSpec:
    label: str
    executable_name: str


SUPPORTED_TARGETS: dict[str, TargetSpec] = {
    "linux-x64": TargetSpec(label="linux-x64", executable_name="tsf"),
    "darwin-x64": TargetSpec(label="darwin-x64", executable_name="tsf"),
    "darwin-arm64": TargetSpec(label="darwin-arm64", executable_name="tsf"),
    "win32-x64": TargetSpec(label="win32-x64", executable_name="tsf.exe"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a bundled TorchShapeFlow CLI executable for the current host.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the bundled executable will be written.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory under which a <target>/ directory will be created.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Label for the host target. Must match the current host platform/arch if provided.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous PyInstaller work directories before building.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a basic bundled-CLI smoke test after building.",
    )
    parser.add_argument(
        "--smoke-test-file",
        type=Path,
        default=Path("tests/fixtures/attention_scores.py"),
        help="Fixture file to analyze during the bundled-CLI smoke test.",
    )
    return parser.parse_args()


def host_target() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        platform_name = "darwin"
    elif system == "linux":
        platform_name = "linux"
    elif system == "windows":
        platform_name = "win32"
    else:
        raise RuntimeError(f"Unsupported platform for bundled CLI build: {platform.system()}")

    if machine in {"x86_64", "amd64"}:
        arch = "x64"
    elif machine in {"arm64", "aarch64"}:
        arch = "arm64"
    else:
        raise RuntimeError(f"Unsupported architecture for bundled CLI build: {platform.machine()}")

    return f"{platform_name}-{arch}"


def resolve_target(requested_target: str | None) -> TargetSpec:
    host = host_target()
    target_label = requested_target or host
    if target_label not in SUPPORTED_TARGETS:
        supported = ", ".join(sorted(SUPPORTED_TARGETS))
        raise RuntimeError(f"Unsupported target '{target_label}'. Supported targets: {supported}")
    if target_label != host:
        raise RuntimeError(
            f"Requested target '{target_label}' does not match current host '{host}'."
        )
    return SUPPORTED_TARGETS[target_label]


def smoke_test_executable(executable_path: Path, repo_root: Path, fixture_file: Path) -> None:
    resolved_fixture = fixture_file.resolve()
    if not resolved_fixture.exists():
        raise RuntimeError(f"Smoke test fixture does not exist: {resolved_fixture}")

    subprocess.run([str(executable_path), "version"], check=True, cwd=repo_root)
    result = subprocess.run(
        [str(executable_path), "check", str(resolved_fixture), "--json"],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    files = payload.get("files", [])
    if not any(
        Path(file_report.get("path", "")).resolve() == resolved_fixture for file_report in files
    ):
        raise RuntimeError(
            f"Bundled CLI smoke test did not report the expected fixture file: {resolved_fixture}"
        )


def main() -> None:
    args = parse_args()
    target = resolve_target(args.target)
    if args.output_dir is None and args.output_root is None:
        raise RuntimeError("Either --output-dir or --output-root must be provided.")
    if args.output_dir is not None and args.output_root is not None:
        raise RuntimeError("Use either --output-dir or --output-root, not both.")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (args.output_root.resolve() / target.label)
    )
    build_root = repo_root / "build" / "bundled-cli" / target.label
    work_dir = build_root / "work"
    spec_dir = build_root / "spec"
    entry_script = repo_root / "scripts" / "tsf_entry.py"

    if args.clean and build_root.exists():
        shutil.rmtree(build_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "tsf",
        "--distpath",
        str(output_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--paths",
        str(repo_root / "src"),
        str(entry_script),
    ]
    subprocess.run(command, check=True, cwd=repo_root)

    executable_path = output_dir / target.executable_name
    if not executable_path.exists():
        raise RuntimeError(f"Bundled executable was not produced at {executable_path}")
    if not target.label.startswith("win32-"):
        executable_path.chmod(0o755)
    if args.smoke_test:
        fixture_file = repo_root / args.smoke_test_file
        smoke_test_executable(executable_path, repo_root, fixture_file)


if __name__ == "__main__":
    main()

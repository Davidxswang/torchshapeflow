# Releasing TorchShapeFlow

This document describes the release flow for:

- the Python package on PyPI
- the VS Code extension as a `.vsix`
- optional marketplace publishing for VS Code Marketplace and Open VSX

## Prerequisites

Before using the automated release workflow, configure these GitHub repository settings.

### PyPI

The release workflow uses trusted publishing.

Required setup:

1. Create the `torchshapeflow` project on PyPI.
2. In PyPI, configure the GitHub repository as a trusted publisher.
3. In GitHub, ensure the `pypi` environment exists if you want environment protection on publish jobs.

No PyPI API token is required when trusted publishing is configured correctly.

### VS Code Marketplace

Optional.

If you want the release workflow to publish the extension to the VS Code Marketplace, add this GitHub secret:

- `VSCE_PAT`

This should be a Personal Access Token created for extension publishing.

### Open VSX

Optional.

If you want the release workflow to publish the extension to Open VSX, add this GitHub secret:

- `OVSX_PAT`

If this secret is missing, the release workflow will skip Open VSX publishing but still package the `.vsix`.

## Local Release Commands

Common local commands:

```bash
make check
make python-dist
make bundle-cli
make extension-package
make extension-package-bundled
make build
```

Version bump commands:

```bash
make bump-patch
make bump-minor
make bump-major
```

These commands update:

- `pyproject.toml`
- `src/torchshapeflow/_version.py`
- `extensions/vscode/package.json`
- `extensions/vscode/package-lock.json`
- `uv.lock` (via `uv lock`)

## Release Procedure

### Test release (TestPyPI)

Use a tag containing `-rc` or `-test` to publish to TestPyPI only:

```bash
git tag vX.Y.Z-rc1
git push origin vX.Y.Z-rc1
```

Install from TestPyPI to verify:

```bash
pip install --index-url https://test.pypi.org/simple/ torchshapeflow
```

### Production release (PyPI)

Once satisfied, bump the version, commit, and push a clean semver tag:

```bash
make check
make bump-patch   # or bump-minor / bump-major
git add pyproject.toml src/torchshapeflow/_version.py extensions/vscode/package.json extensions/vscode/package-lock.json uv.lock
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

Pushing a clean `vX.Y.Z` tag (no `-rc` or `-test` suffix) triggers the full release workflow.

## What the Release Workflow Does

On a clean `vX.Y.Z` tag, `release.yml` will:

1. Build Python artifacts (wheel + sdist).
2. Build bundled `tsf` executables for the supported extension targets.
3. Smoke-test each bundled executable on its build host.
4. Assemble and package one universal VS Code extension (`.vsix`) containing those bundled executables.
5. Publish to PyPI.
6. Optionally publish to the VS Code Marketplace if `VSCE_PAT` exists.
7. Optionally publish to Open VSX if `OVSX_PAT` exists.
8. Create a GitHub release with all artifacts attached.

## Artifact Locations

Local artifact locations:

- Python artifacts: `dist/`
- Bundled CLI artifacts during local/CI builds: `extensions/vscode/bin/<target>/`
- Extension artifact: `extensions/vscode/dist/`

Current bundled extension targets:

- `linux-x64` (built in manylinux2014 for broader glibc compatibility)
- `darwin-arm64`
- `win32-x64`

Expected outputs:

- `dist/torchshapeflow-<version>-py3-none-any.whl`
- `dist/torchshapeflow-<version>.tar.gz`
- `extensions/vscode/dist/torchshapeflow.vsix`

## Notes

- If marketplace secrets are not configured, the workflow still succeeds for packaging and GitHub release creation.
- If PyPI trusted publishing is not configured correctly, the PyPI publish job will fail.

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
make extension-package
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
git tag v0.1.1-rc1
git push origin v0.1.1-rc1
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
git commit -m "Release v0.1.1"
git tag v0.1.1
git push origin main --tags
```

Pushing a clean `vX.Y.Z` tag (no `-rc` or `-test` suffix) triggers the full release workflow.

## What the Release Workflow Does

On a tag like `v0.1.1`, `release.yml` will:

1. Build Python artifacts (wheel + sdist).
2. Build and package the VS Code extension (`.vsix`).
3. Publish to PyPI.
4. Optionally publish to the VS Code Marketplace if `VSCE_PAT` exists.
5. Optionally publish to Open VSX if `OVSX_PAT` exists.
6. Create a GitHub release with all artifacts attached.

## Artifact Locations

Local artifact locations:

- Python artifacts: `dist/`
- Extension artifact: `extensions/vscode/dist/`

Expected outputs:

- `dist/torchshapeflow-<version>-py3-none-any.whl`
- `dist/torchshapeflow-<version>.tar.gz`
- `extensions/vscode/dist/torchshapeflow.vsix`

## Notes

- If marketplace secrets are not configured, the workflow still succeeds for packaging and GitHub release creation.
- If PyPI trusted publishing is not configured correctly, the PyPI publish job will fail.
- The extension package currently uses `--allow-missing-repository`; adding the final public repository URL to the extension manifest later would remove that exception.

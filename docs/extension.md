# VS Code / Cursor Extension

The extension provides inline shape diagnostics and hover information for tensor variables directly in your editor.

## How it works

The extension is CLI-backed: on each file save (or on demand), it runs `tsf check --json` against the active Python file, then surfaces:

- **Diagnostics** — red underlines at the location of shape errors with human-readable messages.
- **Hover shapes** — hover over a tensor variable to see its inferred shape (e.g. `[B, 12, T, 64]`), or hover over a function name to see the full shape signature of its tensor parameters and return value.

The extension does not run a background language server. Each check is a fresh `tsf` invocation against the current file.

## Installing

Search for **Torch Shape Flow** in the VS Code or Cursor Extensions panel and click Install.

Alternatively, download the `.vsix` from the [GitHub Releases](https://github.com/Davidxswang/torchshapeflow/releases) page and install manually:

```
Extensions panel → ⋯ (More Actions) → Install from VSIX...
```

## Requirements

The extension requires the `torchshapeflow` Python package:

```bash
pip install torchshapeflow
```

The extension looks for `tsf` in this order:

1. `.venv/bin/tsf` in the workspace root — picked up automatically if you use a local virtual environment
2. The path in `torchShapeFlow.cliPath` (see Settings below)
3. `tsf` on your system `PATH`

The extension requires VS Code ≥ 1.90 or a compatible Cursor version.

## Building locally

```bash
make extension-build     # development build (faster, no .vsix)
make extension-package   # produces extensions/vscode/dist/torchshapeflow.vsix
```

Requires Node.js ≥ 24 and `npm`.

## Release and marketplace publishing

Triggered automatically by a `v*` tag push via `.github/workflows/release.yml`:

- The `.vsix` is always built and attached to the GitHub Release.
- If the `VSCE_PAT` GitHub Actions secret is set, the extension is published to the VS Code Marketplace.
- If the `OVSX_PAT` secret is set, it is also published to Open VSX.

Both secrets are optional. If missing, the workflow still succeeds — packaging and GitHub Release creation always run. See `RELEASING.md` in the repository root for the full release steps.

## Known limitations

- Hover information is most accurate for annotated parameters and variables assigned in the same function body. References to shapes defined in other files are not resolved.
- Only the active file is analyzed. Cross-file shape propagation is not supported.
- Hover results for re-used variable names reflect the first assignment in the function. Rebinding the same name produces only one hover entry.

## Future direction

- Analysis result caching for faster repeated checks on large files
- A long-running backend or LSP-style service when the CLI-backed model becomes a bottleneck
- Richer reference-aware hover based on a project-level analysis index

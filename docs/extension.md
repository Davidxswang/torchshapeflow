# VS Code / Cursor Extension

The extension provides inline shape diagnostics and hover information for tensor variables directly in your editor.

## How it works

The extension is CLI-backed: on each file save (or on demand), it runs `tsf check --json` against the active Python file, then surfaces:

- **Diagnostics** — red underlines at the location of shape errors with human-readable messages.
- **Hover shapes** — when you hover over a tensor variable, the inferred shape is shown (e.g. `Tensor[B, 12, T, 64]`).

The extension does not run a background language server. Each check is a fresh `tsf` invocation against the current file.

## Installing

The `.vsix` file is attached to each GitHub release under Assets. Install it manually:

```
Extensions panel → ⋯ (More Actions) → Install from VSIX...
```

Once the extension is published to the VS Code Marketplace, it will be installable directly from the Extensions search panel.

## Requirements

- `tsf` must be available on `PATH`, or configured in the extension settings.
- The extension requires VS Code ≥ 1.85 or a compatible Cursor version.

## Building locally

```bash
make extension-build     # development build (faster, no .vsix)
make extension-package   # produces extensions/vscode/dist/torch-shape-flow.vsix
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

# VS Code / Cursor Extension

The extension provides inline shape diagnostics and hover information directly
in your editor. It is designed to make TorchShapeFlow's annotation-first,
symbolic-first workflow usable without leaving VS Code or Cursor.

## How it works

The extension is CLI-backed: when a Python file is opened, saved, or checked on
demand, it runs `tsf check --json` against the active file, then surfaces:

- **Diagnostics** — red underlines at the location of shape errors with human-readable messages.
- **Hover shapes** — hover over a tensor variable to see its inferred shape (e.g. `[B, 12, T, 64]`), hover over a shape alias to see the aliased contract, or hover over a function name to see the full shape signature of its tensor parameters and return value.

The extension does not run a background language server. Each check is a fresh
`tsf` invocation against the current file.

## Installing

Search for **Torch Shape Flow** in the VS Code or Cursor Extensions panel and click Install.

Alternatively, download the `.vsix` from the [GitHub Releases](https://github.com/Davidxswang/torchshapeflow/releases) page and install manually:

```
Extensions panel → ⋯ (More Actions) → Install from VSIX...
```

## Requirements

The published extension bundles `tsf` executables for its supported release
targets, so a separate Python package install is not required for normal use.
Current bundled targets are Linux x64, macOS arm64, and Windows x64.
Other platforms can still use a workspace `.venv`, `torchShapeFlow.cliPath`, or
`tsf` on `PATH`.

The extension looks for `tsf` in this order:

1. The path in `torchShapeFlow.cliPath` (see Settings below), if set
2. `.venv/bin/tsf` or `.venv/Scripts/tsf.exe` in the workspace root
3. The bundled executable shipped with the extension
4. `tsf` on your system `PATH`

The extension requires VS Code ≥ 1.90 or a compatible Cursor version.

## Building locally

```bash
make extension-build            # development build (faster, no .vsix)
make bundle-cli                 # build a bundled CLI for the current host
make extension-package          # package the current extension state into a .vsix
make extension-package-bundled  # build the host bundled CLI, then package the .vsix
```

Requires Node.js ≥ 24 and `npm`.

## Release and marketplace publishing

Triggered automatically by a `v*` tag push via `.github/workflows/release.yml`:

- The `.vsix` is always built and attached to the GitHub Release.
- If the `VSCE_PAT` GitHub Actions secret is set, the extension is published to the VS Code Marketplace.
- If the `OVSX_PAT` secret is set, it is also published to Open VSX.
- Release CI first builds bundled `tsf` executables for the supported targets,
  smoke-tests each bundled binary, then assembles them into one universal `.vsix`.

Both secrets are optional. If missing, the workflow still succeeds — packaging
and GitHub Release creation always run. See [Releasing](releasing.md) for the
full release steps.

## Known limitations

- Diagnostics and hovers are produced only for the active file.
- Imported shape aliases and annotated helper functions in project-local files
  can still affect inference for the active file, because the CLI builds a
  project index during each check.
- Local shape aliases and annotated local variables are reflected in diagnostics
  and hovers, but only within the active file's analysis pass.
- There is no workspace-wide background analysis or long-lived project state;
  results are recomputed on each run.

## Future direction

- Analysis result caching for faster repeated checks on large files
- A long-running backend or LSP-style service when the CLI-backed model becomes a bottleneck
- Richer reference-aware hover based on a project-level analysis index

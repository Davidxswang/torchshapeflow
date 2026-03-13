# Torch Shape Flow VS Code Extension

This extension shells out to `tsf check --json` and turns analyzer output into:

- editor diagnostics
- hover shape information

## Current behavior

- runs on the active Python file
- can run automatically on save
- can be triggered manually with the `Torch Shape Flow: Run Analysis` command
- caches the latest analyzer result per file for hover lookup
- can be packaged locally with `npm run package`

## Requirements

- `tsf` must be available on your `PATH`, or configured explicitly in the extension settings
- the Python project should be analyzable by the TorchShapeFlow CLI

## Release publishing

GitHub Actions can publish this extension conditionally:

- `VSCE_PAT` enables publishing to the VS Code Marketplace
- `OVSX_PAT` enables publishing to Open VSX

If those secrets are not configured, release workflows still package `dist/torch-shape-flow.vsix` and attach it to the GitHub release.

## Notes

The current hover implementation uses analyzer-reported symbol facts and a name-based fallback, so it is intentionally lightweight. A richer editor experience can be built later on top of the same JSON contract or a future long-running backend.

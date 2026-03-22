# Development

## Setup

```bash
git clone https://github.com/Davidxswang/torchshapeflow
cd torchshapeflow
make install   # uv sync --extra dev
```

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/). Node.js ≥ 24 is required only for VS Code extension work.

## Daily workflow

```bash
make check     # format + lint + typecheck + tests  ← run before every PR
make test      # tests only: uv run pytest -q  (xdist parallel by default)
make format    # ruff format .
make lint      # ruff check . --fix
make typecheck # mypy .
```

`make check` runs exactly what CI runs. If it passes locally it will pass in CI.

## Design discipline

- Prefer a single source of truth in both code and docs.
- Factor shared implementation instead of copying analyzer/index logic in two places.
- Put canonical user-facing operator semantics in [Supported Operators](operators.md).
- Use other docs to explain architecture, workflow, or context, then link back to the canonical page instead of restating long support lists.

## Full make target reference

| Target | Command | Purpose |
|---|---|---|
| `install` | `uv sync --extra dev` | Install all dependencies including dev extras |
| `format` | `ruff format .` | Auto-format source |
| `lint` | `ruff check . --fix` | Lint and auto-fix |
| `typecheck` | `mypy .` | Type-check with strict mypy |
| `test` | `uv run pytest -q` | Run test suite (pytest-xdist parallel by default) |
| `check` | format + lint + typecheck + test | Full local CI pass |
| `docs` | `mkdocs build` | Build documentation site into `site/` |
| `docs-serve` | `mkdocs serve` | Serve docs locally at `localhost:8000` |
| `python-dist` | `uv build` | Build wheel and sdist into `dist/` |
| `bundle-cli` | `scripts/build_bundled_cli.py` | Build a bundled `tsf` executable for the current host into `extensions/vscode/bin/<target>/` |
| `extension-build` | `npm run build` in `extensions/vscode` | Development build of the VS Code extension |
| `extension-package` | `bundle-cli` + `npm run package` in `extensions/vscode` | Rebuild the host bundled CLI, then package `.vsix` into `extensions/vscode/dist/` |
| `build` | `python-dist` + `extension-package` | Build all release artifacts |
| `bump-patch` | `scripts/bump_version.py patch` + `uv lock` | Bump patch version across all version files and lock |
| `bump-minor` | `scripts/bump_version.py minor` + `uv lock` | Bump minor version |
| `bump-major` | `scripts/bump_version.py major` + `uv lock` | Bump major version |
| `clean` | `rm -rf ...` | Remove build and cache artifacts |

## CI workflows

All workflows live in `.github/workflows/`.

### `ci.yml` — every push and pull request

Four job groups run in parallel:

- **`check`** (runs once on Python 3.12): format check, lint, type-check, docs build, bundled CLI build for the host runner, and VS Code extension packaging.
- **`bundled-cli-linux`** (manylinux2014 x86_64): builds the Linux x64 bundled CLI in an older glibc environment and smoke-tests the binary.
- **`bundled-cli`** (matrix: macOS arm64, Windows x64): builds and smoke-tests the non-Linux bundled CLI targets.
- **`test`** (matrix: Python 3.10–3.14 × Ubuntu, macOS, Windows): `pytest -q` with xdist parallelism enabled by the repo pytest config.

### `docs.yml` — push to `main` and manual trigger

Builds the MkDocs site and deploys it to GitHub Pages.

### `build-artifacts.yml` — push to `main`, PR, and manual trigger

Builds and uploads:

- `python-dist` — wheel + sdist from `uv build`
- `bundled-cli-<target>` — bundled `tsf` executables for the extension release targets
- `vscode-extension` — `.vsix` from `npm run package`

### `release.yml` — triggered by a `v*` tag push

1. Builds Python artifacts (wheel + sdist)
2. Builds bundled `tsf` executables for the supported extension targets
   - Linux x64 is built in a manylinux2014 container for broader glibc compatibility
   - Each bundled binary is smoke-tested before upload
3. Assembles and packages one universal VS Code extension (`.vsix`)
4. Publishes to PyPI using trusted publishing (no API token required)
5. Optionally publishes to VS Code Marketplace if `VSCE_PAT` secret is set
6. Optionally publishes to Open VSX if `OVSX_PAT` secret is set
7. Creates a GitHub Release with all artifacts attached

See [Releasing](releasing.md) for the full release procedure including version bump steps.

## Adding a new operator

1. **Add an inference function** to the appropriate module in `src/torchshapeflow/rules/`:
   - Shape transformations (reshape, permute, etc.) → `shape_ops.py`
   - New `nn.*` module type → new file, e.g. `embedding.py`

2. **Follow the function contract:**
   - Return `TensorValue` on success
   - Return `None` if inference is not possible — never raise
   - Add a docstring with explicit shape signatures: `shape: (Batch, Channels, Height, Width)`

3. **Export from `src/torchshapeflow/rules/__init__.py`.**

4. **Wire dispatch in `src/torchshapeflow/analyzer.py`:**
   - Tensor methods → `_eval_tensor_method`
   - `torch.*` functions → `_eval_call`
   - `nn.*` modules → `_parse_module_spec` (constructor) + `_collect_class_specs` + `_eval_call` (forward call)

5. **Add tests.** Every operator needs at minimum:
   - One valid case with a shape assertion
   - One invalid case (wrong dims) verifying `None` is returned
   - Edge cases: negative indices, symbolic dims, boundary conditions

6. Run `make check` to verify.

## Test conventions

- Test helpers use `_t(*dims: int | str)`: `int` → `ConstantDim`, `str` → `SymbolicDim`.
- Rule unit tests go in `tests/test_shape_ops.py` or `tests/test_rules.py`.
- End-to-end analyzer tests go in `tests/test_analyzer.py`.
- Always assert `result is not None` before asserting on `str(result.shape)`.

## Version management

Version is kept in sync across five files by `scripts/bump_version.py` + `uv lock`:

- `pyproject.toml`
- `src/torchshapeflow/_version.py`
- `extensions/vscode/package.json`
- `extensions/vscode/package-lock.json`
- `uv.lock`

Always use `make bump-patch / bump-minor / bump-major`. Never edit version files by hand.

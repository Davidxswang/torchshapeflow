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
make test      # tests only: uv run pytest -q
make format    # ruff format .
make lint      # ruff check . --fix
make typecheck # mypy .
```

`make check` runs exactly what CI runs. If it passes locally it will pass in CI.

## Full make target reference

| Target | Command | Purpose |
|---|---|---|
| `install` | `uv sync --extra dev` | Install all dependencies including dev extras |
| `format` | `ruff format .` | Auto-format source |
| `lint` | `ruff check . --fix` | Lint and auto-fix |
| `typecheck` | `mypy .` | Type-check with strict mypy |
| `test` | `uv run pytest -q` | Run test suite |
| `check` | format + lint + typecheck + test | Full local CI pass |
| `docs` | `mkdocs build` | Build documentation site into `site/` |
| `docs-serve` | `mkdocs serve` | Serve docs locally at `localhost:8000` |
| `python-dist` | `uv build` | Build wheel and sdist into `dist/` |
| `extension-build` | `npm run build` in `extensions/vscode` | Development build of the VS Code extension |
| `extension-package` | `npm run package` in `extensions/vscode` | Package `.vsix` into `extensions/vscode/dist/` |
| `build` | `python-dist` + `extension-package` | Build all release artifacts |
| `bump-patch` | `scripts/bump_version.py patch` + `uv lock` | Bump patch version across all version files and lock |
| `bump-minor` | `scripts/bump_version.py minor` + `uv lock` | Bump minor version |
| `bump-major` | `scripts/bump_version.py major` + `uv lock` | Bump major version |
| `clean` | `rm -rf ...` | Remove build and cache artifacts |

## CI workflows

All workflows live in `.github/workflows/`.

### `ci.yml` — every push and pull request

Two jobs run in parallel:

- **`check`** (runs once on Python 3.12): format check, lint, type-check, docs build, VS Code extension build.
- **`test`** (matrix: Python 3.10, 3.11, 3.12, 3.13): `pytest -q`.

### `docs.yml` — push to `main` and manual trigger

Builds the MkDocs site and deploys it to GitHub Pages.

### `build-artifacts.yml` — push to `main`, PR, and manual trigger

Builds and uploads two artifacts:

- `python-dist` — wheel + sdist from `uv build`
- `vscode-extension` — `.vsix` from `npm run package`

### `release.yml` — triggered by a `v*` tag push

1. Builds Python artifacts (wheel + sdist) and the VS Code extension (`.vsix`)
2. Publishes to PyPI using trusted publishing (no API token required)
3. Optionally publishes to VS Code Marketplace if `VSCE_PAT` secret is set
4. Optionally publishes to Open VSX if `OVSX_PAT` secret is set
5. Creates a GitHub Release with all artifacts attached

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

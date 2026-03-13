# TorchShapeFlow — Agent Guide

TorchShapeFlow is a static, AST-based shape analyzer for PyTorch. No runtime dependency on torch.
The canonical design document is this file plus `docs/`. `RELEASING.md` covers the release procedure.

## Repo map

| Path | Purpose |
|---|---|
| `src/torchshapeflow/model.py` | Core Dim types, TensorShape, TensorValue, shape arithmetic |
| `src/torchshapeflow/analyzer.py` | AST walker; builds shape environment, emits diagnostics |
| `src/torchshapeflow/parser.py` | Parses `Annotated[Tensor, Shape(...)]` annotation nodes |
| `src/torchshapeflow/rules/` | Shape inference functions (one concern per module) |
| `src/torchshapeflow/diagnostics.py` | Diagnostic dataclass and Severity type |
| `src/torchshapeflow/report.py` | FileReport and HoverFact |
| `src/torchshapeflow/cli.py` | Typer CLI (`tsf check`, `tsf version`) |
| `src/torchshapeflow/annotations.py` | Public `Shape` class |
| `tests/` | Pytest test suite |
| `examples/` | Runnable examples for manual and CI testing |
| `docs/` | Documentation site (MkDocs) — system of record |
| `extensions/vscode/` | VS Code / Cursor extension (TypeScript, CLI-backed) |
| `scripts/bump_version.py` | Keeps version in sync across all four version files |
| `.github/workflows/` | CI, docs deploy, build artifacts, release |
| `Makefile` | All standard development commands |
| `RELEASING.md` | Full release procedure |

## Key commands

```bash
make install       # uv sync --extra dev
make check         # format + lint + typecheck + tests  ← run before every PR
make test          # uv run pytest -q
make lint          # ruff check . --fix
make format        # ruff format .
make typecheck     # mypy .
make docs          # mkdocs build
make docs-serve    # mkdocs serve  (localhost:8000)
make build         # python-dist + extension-package
make bump-patch    # bump patch version across all version files
make bump-minor    # bump minor version
make bump-major    # bump major version
make clean         # remove build and cache artifacts
```

## Documentation — where to look

| Topic | Location |
|---|---|
| Development workflow, all make targets, CI, how to add operators | `docs/development.md` |
| Module map, analysis pipeline, Dim types, shape environment | `docs/architecture.md` |
| Every supported operator with shape signatures | `docs/operators.md` |
| Annotation syntax (`Shape`, string vs int dims, TypeAlias) | `docs/syntax.md` |
| Known limitations and non-goals | `docs/limitations.md` |
| VS Code / Cursor extension | `docs/extension.md` |
| Release procedure | `RELEASING.md` |

## Engineering principles

- One concern per module. New operators go in `src/torchshapeflow/rules/`.
- Every supported operator needs tests before it ships.
- Inference functions return `None` on failure — never raise.
- Diagnostics use stable codes: `TSF1001`, `TSF1002`, ...
- `make check` must pass before a PR is merged (it runs exactly what CI runs).
- No `Any` without justification. `mypy --strict` must pass.
- Only direct `self.attr` access is tracked; aliases (`m = self.linear`) are not.

## Conventions

- Python ≥ 3.10. All files start with `from __future__ import annotations`.
- Typed throughout. Use `pathlib.Path` for all file operations.
- Tensor shape docstrings state shapes explicitly: `shape: (Batch, Channels, Height, Width)`.
- Test helpers use `_t(*dims: int | str)`: `int` → `ConstantDim`, `str` → `SymbolicDim`.
- Version is managed by `scripts/bump_version.py`. Never edit version files by hand.

## Decision policy

- Prefer the smallest design that keeps extension paths open.
- Prefer real operator coverage (from actual PyTorch repos) over speculative abstractions.
- When inference is not possible, return `None` and emit a diagnostic — never guess.
- Boring, explicit, composable code is easier for agents to reason about than clever abstractions.

PYTHON ?= python

.PHONY: install format lint typecheck test check docs docs-serve build python-dist bundle-cli extension-build extension-package extension-package-bundled bump-patch bump-minor bump-major clean

install:
	uv sync --extra dev

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

typecheck:
	uv run mypy .

test:
	uv run pytest -q

check: format lint typecheck test

docs:
	uv run mkdocs build

docs-serve:
	uv run mkdocs serve

build: python-dist extension-package-bundled

python-dist:
	uv build

bundle-cli:
	uv run python scripts/build_bundled_cli.py --output-root extensions/vscode/bin --clean --smoke-test

extension-build:
	cd extensions/vscode && npm ci && npm run build

extension-package:
	cd extensions/vscode && npm ci && npm run package

extension-package-bundled: bundle-cli extension-package

bump-patch:
	uv run python scripts/bump_version.py patch
	uv lock

bump-minor:
	uv run python scripts/bump_version.py minor
	uv lock

bump-major:
	uv run python scripts/bump_version.py major
	uv lock

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist site extensions/vscode/bin extensions/vscode/dist extensions/vscode/out

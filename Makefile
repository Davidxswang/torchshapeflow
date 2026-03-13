PYTHON ?= python

.PHONY: install format lint typecheck test check docs docs-serve build python-dist extension-build extension-package bump-patch bump-minor bump-major clean

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

build: python-dist extension-package

python-dist:
	uv build

extension-build:
	cd extensions/vscode && npm ci && npm run build

extension-package:
	cd extensions/vscode && npm ci && npm run package

bump-patch:
	uv run python scripts/bump_version.py patch

bump-minor:
	uv run python scripts/bump_version.py minor

bump-major:
	uv run python scripts/bump_version.py major

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist site extensions/vscode/dist extensions/vscode/out

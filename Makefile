PYTHON ?= python

.PHONY: help install format lint typecheck test check docs docs-serve build python-dist bundle-cli extension-build extension-package bump-patch bump-minor bump-major clean

help:
	@printf '%s\n' \
		'Common targets:' \
		'  make install                    Install dev dependencies' \
		'  make check                      Format, lint, typecheck, and test' \
		'  make build                      Build Python artifacts and a bundled VSIX' \
		'' \
		'Extension targets:' \
		'  make extension-build            Build extension JS only (fast dev loop)' \
		'  make bundle-cli                 Rebuild bundled tsf for this host' \
		'  make extension-package          Rebuild bundled tsf, then package VSIX' \
		'' \
		'Version targets:' \
		'  make bump-patch | bump-minor | bump-major'

install:
	uv sync --extra dev

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

typecheck:
	uv run ty check

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

bundle-cli:
	uv run python scripts/build_bundled_cli.py --output-root extensions/vscode/bin --clean --smoke-test

extension-build:
	cd extensions/vscode && npm ci && npm run build

extension-package: bundle-cli
	cd extensions/vscode && npm ci && npm run package

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
	rm -rf .pytest_cache .mypy_cache .ruff_cache .ty_cache build dist site extensions/vscode/bin extensions/vscode/dist extensions/vscode/out

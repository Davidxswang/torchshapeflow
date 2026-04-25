"""TSF analyzer tests against the realistic ``tests/interop/fixtures/`` files.

These fixtures are the same ones used by the type-checker interop tests, but
exercised through TSF directly. The contract:

- Clean fixtures must produce zero error-severity diagnostics.
- Bug fixtures must produce the documented TSF code (and only that one).
- Hover counts are bounded — the analyzer should produce at least the
  signature hover plus parameter hovers for each annotated function.

Catches analyzer regressions on realistic shape patterns (transformer block,
ResCNN, TypeAlias vocabulary, common shape-bug shapes) — a stronger signal
than the synthetic toy examples in ``tests/test_analyzer.py``.
"""

from __future__ import annotations

import pytest

from torchshapeflow.analyzer import analyze_path

from .conftest import (
    CLEAN_FIXTURES,
    fixture_path,
)


@pytest.mark.parametrize("fixture", CLEAN_FIXTURES)
def test_clean_fixture_has_no_errors(fixture: str) -> None:
    """Each clean fixture must analyze with zero error-severity diagnostics."""
    report = analyze_path(fixture_path(fixture))
    errors = [d for d in report.diagnostics if d.severity == "error"]
    assert errors == [], f"{fixture} produced unexpected errors: {errors}"


@pytest.mark.parametrize("fixture", CLEAN_FIXTURES)
def test_clean_fixture_has_signature_hovers(fixture: str) -> None:
    """Every clean fixture annotates at least one function — assert TSF emits the
    matching ``signature`` hover so the agent can render the function shape."""
    report = analyze_path(fixture_path(fixture))
    signatures = [h for h in report.hovers if h.kind == "signature"]
    assert signatures, f"{fixture} produced no signature hovers"


def test_bug_linear_emits_tsf1007_with_structured_fields() -> None:
    """The Linear-mismatch fixture must trigger TSF1007 with the documented
    enriched fields (``expected`` / ``actual`` / ``hint``)."""
    report = analyze_path(fixture_path("bug_linear_in_features.py"))
    errors = [d for d in report.diagnostics if d.severity == "error"]
    assert len(errors) == 1
    diag = errors[0]
    assert diag.code == "TSF1007"
    assert diag.expected == "last dim = 768"
    assert diag.actual is not None and "512" in diag.actual
    assert diag.hint is not None and "768" in diag.hint and "512" in diag.hint


def test_bug_matmul_emits_tsf1003_with_structured_fields() -> None:
    """The matmul inner-dim fixture must trigger TSF1003 with the documented
    enriched fields and reference the correct shapes."""
    report = analyze_path(fixture_path("bug_matmul_inner.py"))
    errors = [d for d in report.diagnostics if d.severity == "error"]
    assert len(errors) == 1
    diag = errors[0]
    assert diag.code == "TSF1003"
    assert diag.expected is not None and "last dim of left" in diag.expected
    assert diag.actual is not None
    assert "[B, H, T, 64]" in diag.actual
    assert "[B, H, 128, T]" in diag.actual
    assert diag.hint is not None and "transpose" in diag.hint


def test_typealias_pattern_resolves_aliases_in_hovers() -> None:
    """The TypeAlias fixture should produce ``alias``-kind hovers for the
    declared aliases — agents and editors rely on these to render the
    aliased shape on hover."""
    report = analyze_path(fixture_path("clean_typealias.py"))
    alias_names = {h.name for h in report.hovers if h.kind == "alias"}
    assert {"TokenSequence", "Embedding"}.issubset(alias_names)

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    path: Path
    line: int
    column: int
    severity: Severity = "error"
    end_line: int | None = None
    end_column: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Optional structured fields for machine consumption. When populated, they
    # are the source of truth and `message` is rendered from them (via
    # render_message). Unset fields are omitted from JSON output.
    expected: str | None = None
    actual: str | None = None
    hint: str | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "path": str(self.path),
            "line": self.line,
            "column": self.column,
        }
        if self.end_line is not None:
            data["end_line"] = self.end_line
        if self.end_column is not None:
            data["end_column"] = self.end_column
        if self.notes:
            data["notes"] = list(self.notes)
        if self.expected is not None:
            data["expected"] = self.expected
        if self.actual is not None:
            data["actual"] = self.actual
        if self.hint is not None:
            data["hint"] = self.hint
        return data


def render_message(
    summary: str,
    *,
    expected: str | None = None,
    actual: str | None = None,
    hint: str | None = None,
) -> str:
    """Render the human-readable message from a summary plus structured fields.

    Centralizes the format so the prose message and structured fields never
    drift. Used by ModuleContext.shape_error to guarantee consistency.
    """
    parts: list[str] = [summary]
    if expected is not None:
        parts.append(f"expected {expected}")
    if actual is not None:
        parts.append(f"got {actual}")
    if hint is not None:
        parts.append(f"hint: {hint}")
    return " — ".join(parts)

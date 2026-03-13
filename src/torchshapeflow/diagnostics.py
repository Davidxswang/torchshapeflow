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
        return data

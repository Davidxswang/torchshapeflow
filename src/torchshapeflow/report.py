from __future__ import annotations

from dataclasses import dataclass, field

from torchshapeflow.diagnostics import Diagnostic


@dataclass(frozen=True)
class HoverFact:
    line: int
    column: int
    end_line: int
    end_column: int
    name: str
    shape: str

    def to_dict(self) -> dict[str, object]:
        return {
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "name": self.name,
            "shape": self.shape,
        }


@dataclass
class FileReport:
    path: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    hovers: list[HoverFact] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "hovers": [item.to_dict() for item in self.hovers],
        }

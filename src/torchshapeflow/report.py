from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from torchshapeflow.diagnostics import Diagnostic

HoverKind = Literal["value", "signature", "alias"]
SuggestionKind = Literal["return_annotation"]


@dataclass(frozen=True)
class HoverFact:
    line: int
    column: int
    end_line: int
    end_column: int
    name: str
    shape: str
    kind: HoverKind = "value"

    def to_dict(self) -> dict[str, object]:
        return {
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "name": self.name,
            "shape": self.shape,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class Suggestion:
    """An annotation the analyzer proposes but does not apply.

    The user (or an agent) reviews and decides whether to add it to the source.
    TorchShapeFlow never commits these automatically — it is a contract
    verifier, not a shape guesser.
    """

    line: int
    column: int
    end_line: int
    end_column: int
    function: str
    shape: str
    annotation: str
    kind: SuggestionKind = "return_annotation"

    def to_dict(self) -> dict[str, object]:
        return {
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "function": self.function,
            "shape": self.shape,
            "annotation": self.annotation,
            "kind": self.kind,
        }


@dataclass
class FileReport:
    path: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    hovers: list[HoverFact] = field(default_factory=list)
    # `suggestions` is populated by the analyzer but deliberately excluded from
    # ``to_dict()``. The generic report JSON (emitted by ``tsf check --json``
    # and consumed by the VS Code extension / library callers) is intentionally
    # stable and has no need for proposal data. ``tsf suggest`` reads this
    # attribute directly and renders its own payload.
    suggestions: list[Suggestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "hovers": [item.to_dict() for item in self.hovers],
        }

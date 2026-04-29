from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from torchshapeflow.diagnostics import Diagnostic, Severity, render_message
from torchshapeflow.index import FuncSig
from torchshapeflow.model import Dim, TensorValue
from torchshapeflow.report import HoverFact, Suggestion


@dataclass
class ModuleContext:
    path: Path
    diagnostics: list[Diagnostic] = field(default_factory=list)
    hovers: list[HoverFact] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)
    aliases: dict[str, TensorValue] = field(default_factory=dict)
    func_sigs: dict[str, FuncSig] = field(default_factory=dict)
    return_shape: TensorValue | None = None
    collected_returns: list[TensorValue | None] = field(default_factory=list)
    in_annotated_function: bool = False
    # Scalar self.attr values from __init__ (e.g. self.hidden = hidden_dim → SymbolicDim).
    self_scalars: dict[str, Dim] = field(default_factory=dict)
    # Tensor self.attr values captured from __init__ (e.g. register_buffer or direct assignment).
    self_tensors: dict[str, TensorValue] = field(default_factory=dict)
    # Method signatures for the current class being analyzed.
    method_sigs: dict[str, FuncSig] = field(default_factory=dict)

    def error(
        self,
        node: ast.AST,
        code: str,
        message: str,
        severity: Severity = "error",
    ) -> None:
        """Append a diagnostic at the location of *node*.

        Line numbers are 1-based (from ast); column offsets are converted from
        0-based (ast) to 1-based by adding 1.
        """
        self.diagnostics.append(
            Diagnostic(
                code=code,
                message=message,
                path=self.path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                severity=severity,
            )
        )

    def shape_error(
        self,
        node: ast.AST,
        code: str,
        summary: str,
        *,
        expected: str | None = None,
        actual: str | None = None,
        hint: str | None = None,
        severity: Severity = "error",
    ) -> None:
        """Append a shape-mismatch diagnostic with structured fields.

        Structured fields are the source of truth; the human-readable message
        is rendered from them via ``render_message`` to keep prose and data in
        sync. Agents and editors can consume ``expected`` / ``actual`` /
        ``hint`` directly from JSON output.
        """
        self.diagnostics.append(
            Diagnostic(
                code=code,
                message=render_message(summary, expected=expected, actual=actual, hint=hint),
                path=self.path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                severity=severity,
                expected=expected,
                actual=actual,
                hint=hint,
            )
        )

    def hover(self, name: str, node: ast.AST, tensor: TensorValue) -> None:
        self.hovers.append(
            HoverFact(
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                end_column=getattr(node, "end_col_offset", getattr(node, "col_offset", 0)) + 1,
                name=name,
                shape=str(tensor.shape),
            )
        )

    def hover_alias(self, name: str, node: ast.AST, tensor: TensorValue) -> None:
        self.hovers.append(
            HoverFact(
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                end_column=getattr(node, "end_col_offset", getattr(node, "col_offset", 0)) + 1,
                name=name,
                shape=str(tensor.shape),
                kind="alias",
            )
        )

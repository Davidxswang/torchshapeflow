import * as vscode from "vscode";

import type { FileReport } from "./client";

export class DiagnosticsController {
  private readonly collection = vscode.languages.createDiagnosticCollection("torchshapeflow");

  dispose(): void {
    this.collection.dispose();
  }

  update(uri: vscode.Uri, report: FileReport | null): void {
    if (!report) {
      this.collection.delete(uri);
      return;
    }

    const diagnostics = report.diagnostics.map((item) => {
      const start = new vscode.Position(Math.max(item.line - 1, 0), Math.max(item.column - 1, 0));
      const end = new vscode.Position(
        Math.max((item.end_line ?? item.line) - 1, 0),
        Math.max((item.end_column ?? item.column) - 1, 0)
      );
      const diagnostic = new vscode.Diagnostic(
        new vscode.Range(start, end),
        item.message,
        item.severity === "warning"
          ? vscode.DiagnosticSeverity.Warning
          : vscode.DiagnosticSeverity.Error
      );
      diagnostic.code = item.code;
      diagnostic.source = "torchshapeflow";
      return diagnostic;
    });

    this.collection.set(uri, diagnostics);
  }
}

import * as vscode from "vscode";

import type { FileReport, HoverFact } from "./client";

export class HoverController implements vscode.HoverProvider {
  private readonly reports = new Map<string, FileReport>();

  update(uri: vscode.Uri, report: FileReport | null): void {
    if (report) {
      this.reports.set(uri.toString(), report);
      return;
    }
    this.reports.delete(uri.toString());
  }

  async provideHover(
    document: vscode.TextDocument,
    position: vscode.Position
  ): Promise<vscode.Hover | null> {
    const report = this.reports.get(document.uri.toString());
    if (!report) {
      return null;
    }

    const hoverFact = this.findBestHover(report.hovers, position);
    if (!hoverFact) {
      return null;
    }
    const range = new vscode.Range(
      new vscode.Position(Math.max(hoverFact.line - 1, 0), Math.max(hoverFact.column - 1, 0)),
      new vscode.Position(Math.max(hoverFact.end_line - 1, 0), Math.max(hoverFact.end_column - 1, 0))
    );

    const markdown = new vscode.MarkdownString();
    markdown.appendCodeblock(hoverFact.shape, "text");
    const label = hoverFact.shape.startsWith("(")
      ? `Shape signature for \`${hoverFact.name}\`.`
      : `Inferred shape for \`${hoverFact.name}\`.`;
    markdown.appendMarkdown(`\n${label}`);
    return new vscode.Hover(markdown, range);
  }

  private findBestHover(hovers: HoverFact[], position: vscode.Position): HoverFact | undefined {
    return hovers.find((hover) => {
      const startLine = hover.line - 1;
      const endLine = hover.end_line - 1;
      if (position.line < startLine || position.line > endLine) {
        return false;
      }
      const startColumn = hover.column - 1;
      const endColumn = hover.end_column - 1;
      if (position.line === startLine && position.character < startColumn) {
        return false;
      }
      if (position.line === endLine && position.character > endColumn) {
        return false;
      }
      return true;
    });
  }
}

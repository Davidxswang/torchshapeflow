import * as vscode from "vscode";

import { DiagnosticsController } from "./diagnostics";
import { HoverController } from "./hover";
import { TorchShapeFlowClient } from "./client";

export function activate(context: vscode.ExtensionContext): void {
  const client = new TorchShapeFlowClient(context.extensionPath);
  const diagnostics = new DiagnosticsController();
  const hover = new HoverController();
  const output = vscode.window.createOutputChannel("Torch Shape Flow");

  context.subscriptions.push(
    diagnostics,
    output,
    vscode.languages.registerHoverProvider({ language: "python" }, hover)
  );

  const runAnalysis = async (document: vscode.TextDocument): Promise<void> => {
    if (document.languageId !== "python") {
      return;
    }

    try {
      const report = await client.analyzeDocument(document);
      diagnostics.update(document.uri, report);
      hover.update(document.uri, report);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      output.appendLine(message);
      void vscode.window.showErrorMessage(`Torch Shape Flow analysis failed: ${message}`);
    }
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("torchShapeFlow.runAnalysis", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        return;
      }
      await runAnalysis(editor.document);
    }),
    vscode.workspace.onDidSaveTextDocument(async (document) => {
      const runOnSave = vscode.workspace
        .getConfiguration("torchShapeFlow")
        .get<boolean>("runOnSave", true);
      if (runOnSave) {
        await runAnalysis(document);
      }
    }),
    vscode.workspace.onDidOpenTextDocument(async (document) => {
      if (document.languageId === "python") {
        await runAnalysis(document);
      }
    })
  );

  const activeDocument = vscode.window.activeTextEditor?.document;
  if (activeDocument && activeDocument.languageId === "python") {
    void runAnalysis(activeDocument);
  }
}

export function deactivate(): void {}

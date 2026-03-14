import * as vscode from "vscode";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import * as fs from "node:fs";
import * as path from "node:path";

const execFileAsync = promisify(execFile);

export interface AnalyzerDiagnostic {
  code: string;
  severity: "error" | "warning";
  message: string;
  path: string;
  line: number;
  column: number;
  end_line?: number;
  end_column?: number;
}

export interface HoverFact {
  line: number;
  column: number;
  end_line: number;
  end_column: number;
  name: string;
  shape: string;
}

export interface FileReport {
  path: string;
  diagnostics: AnalyzerDiagnostic[];
  hovers: HoverFact[];
}

interface AnalyzerPayload {
  files: FileReport[];
}

export class TorchShapeFlowClient {
  async analyzeDocument(document: vscode.TextDocument): Promise<FileReport | null> {
    if (document.languageId !== "python") {
      return null;
    }

    const cwd = this.getWorkingDirectory(document);
    const cliPath = this.resolveCliPath(cwd);

    try {
      const { stdout } = await execFileAsync(cliPath, ["check", document.uri.fsPath, "--json"], {
        cwd,
      });
      return this.parsePayload(stdout, document.uri);
    } catch (error) {
      const processError = error as NodeJS.ErrnoException & {
        stdout?: string;
        stderr?: string;
      };
      if (processError.stdout) {
        return this.parsePayload(processError.stdout, document.uri);
      }
      throw new Error(processError.stderr || processError.message || "TorchShapeFlow CLI failed.");
    }
  }

  private parsePayload(stdout: string, uri: vscode.Uri): FileReport | null {
    const payload = JSON.parse(stdout) as AnalyzerPayload;
    return payload.files.find((file) => file.path === uri.fsPath) ?? payload.files[0] ?? null;
  }

  private resolveCliPath(cwd: string): string {
    // Prefer .venv/bin/tsf in the workspace root (uv / virtualenv install).
    const venvBin = path.join(cwd, ".venv", "bin", "tsf");
    if (fs.existsSync(venvBin)) {
      return venvBin;
    }
    // Fall back to the user-configured path (or bare "tsf" on PATH).
    return vscode.workspace
      .getConfiguration("torchShapeFlow")
      .get<string>("cliPath", "tsf");
  }

  private getWorkingDirectory(document: vscode.TextDocument): string {
    const folder = vscode.workspace.getWorkspaceFolder(document.uri);
    if (folder) {
      return folder.uri.fsPath;
    }
    return vscode.Uri.joinPath(document.uri, "..").fsPath;
  }
}

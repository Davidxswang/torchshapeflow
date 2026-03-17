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
  kind?: "value" | "signature" | "alias";
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
  constructor(private readonly extensionPath: string) {}

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
    const configuredCli = vscode.workspace
      .getConfiguration("torchShapeFlow")
      .get<string>("cliPath", "")
      .trim();
    if (configuredCli) {
      return configuredCli;
    }

    const workspaceCli = this.resolveWorkspaceCli(cwd);
    if (workspaceCli) {
      return workspaceCli;
    }

    const bundledCli = this.resolveBundledCli();
    if (bundledCli) {
      return bundledCli;
    }

    return "tsf";
  }

  private getWorkingDirectory(document: vscode.TextDocument): string {
    const folder = vscode.workspace.getWorkspaceFolder(document.uri);
    if (folder) {
      return folder.uri.fsPath;
    }
    return vscode.Uri.joinPath(document.uri, "..").fsPath;
  }

  private resolveWorkspaceCli(cwd: string): string | null {
    const venvDir = path.join(cwd, ".venv");
    const executable = process.platform === "win32" ? "tsf.exe" : "tsf";
    const candidate =
      process.platform === "win32"
        ? path.join(venvDir, "Scripts", executable)
        : path.join(venvDir, "bin", executable);
    return fs.existsSync(candidate) ? candidate : null;
  }

  private resolveBundledCli(): string | null {
    const executable = process.platform === "win32" ? "tsf.exe" : "tsf";
    const candidate = path.join(this.extensionPath, "bin", this.hostTarget(), executable);
    if (!fs.existsSync(candidate)) {
      return null;
    }
    if (process.platform !== "win32") {
      try {
        fs.chmodSync(candidate, 0o755);
      } catch {
        // Ignore chmod failures and try to execute the bundled binary as-is.
      }
    }
    return candidate;
  }

  private hostTarget(): string {
    const arch = process.arch === "x64" || process.arch === "arm64" ? process.arch : "unknown";
    return `${process.platform}-${arch}`;
  }
}

# For AI Coding Agents

This page is about TorchShapeFlow's integration with AI coding agents — Claude
Code, Cursor, Aider, Copilot, and any tool-using LLM that edits PyTorch code.
If you are a human looking for a tutorial, read the [Quickstart](quickstart.md).

## Install in Claude Code

Two commands, in order:

```text
/plugin marketplace add Davidxswang/torchshapeflow
/plugin install torchshapeflow@torchshapeflow
```

The first registers this repository as a plugin marketplace (Claude Code pulls
from `main` by default; pin a release with `#v0.7.0`-style refs if you need
reproducibility). The second installs the `torchshapeflow` plugin from that
marketplace. The `@torchshapeflow` suffix disambiguates the plugin name from
the marketplace name (both happen to be `torchshapeflow` here).

The plugin ships three things:

- An **MCP server** with tools `check`, `suggest`, and `hover_at`, launched on
  demand via `uvx` from stdio — no global install, no config-file editing.
- A **skill** that teaches the agent when to reach for TSF, how to interpret
  structured diagnostics, and how to propose annotations. This is the
  canonical agent workflow; see
  [skills/torchshapeflow/SKILL.md](https://github.com/Davidxswang/torchshapeflow/blob/main/skills/torchshapeflow/SKILL.md)
  for the exact text the agent reads.
- A **post-edit hook** that auto-runs `tsf check` after `Write` / `Edit`
  operations on `.py` files and surfaces shape errors to the session.

After install, the agent knows how to use TSF without any further prompting.
The skill description handles when-to-invoke; the hook handles after-edit
sweep; the MCP tools handle on-demand queries.

## Using the MCP server without the plugin

If you're on a runtime that doesn't use Claude Code plugins (Cursor, Aider,
etc.), configure the MCP server directly. Minimal `.mcp.json` entry:

```json
{
  "mcpServers": {
    "torchshapeflow": {
      "command": "uvx",
      "args": ["--from", "torchshapeflow[mcp]", "tsf", "mcp"]
    }
  }
}
```

The three tools — `check`, `suggest`, `hover_at` — are the same as those
surfaced by the plugin. See
[skills/torchshapeflow/SKILL.md](https://github.com/Davidxswang/torchshapeflow/blob/main/skills/torchshapeflow/SKILL.md)
for signatures and usage semantics (single source of truth). The
underlying JSON payloads are documented in
[Architecture — Diagnostic JSON schema](architecture.md#diagnostic-json-schema).

## CLI-only usage (no MCP)

Agents that want to shell out directly instead of going through MCP can invoke
the same commands:

```bash
tsf check --json path/to/file.py     # exit 1 on shape errors
tsf suggest path/to/file.py          # exit 1 iff analysis surfaced errors
```

Both produce the same JSON you would see via MCP. The agent workflow in
[the skill](https://github.com/Davidxswang/torchshapeflow/blob/main/skills/torchshapeflow/SKILL.md)
applies identically.

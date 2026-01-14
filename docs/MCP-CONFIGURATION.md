# QENEX LAB MCP Configuration Guide

This document explains how to configure Model Context Protocol (MCP) servers for QENEX LAB, including both local development and GitHub Copilot coding agent integration.

## Overview

MCP (Model Context Protocol) is an open standard by Anthropic that allows AI models to connect to external tools and data sources. QENEX LAB supports MCP in two contexts:

1. **Local Development** - MCP servers run locally alongside QENEX LAB
2. **GitHub Copilot Coding Agent** - MCP servers provide tools to Copilot on GitHub.com

## Quick Setup

### 1. Local Development Configuration

Create or edit `qenex.json` in your project root:

```json
{
  "$schema": "https://opencode.ai/config.schema.json",
  "mcp": {
    "github": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "environment": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
    }
  }
}
```

Create a `.env` file (never commit this!):

```bash
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_your_actual_token_here
```

### 2. GitHub Copilot Coding Agent Configuration

Navigate to your repository on GitHub.com:

1. Go to **Settings** → **Copilot** → **Coding agent**
2. In the **MCP configuration** field, enter:

```json
{
  "mcpServers": {
    "github": {
      "type": "local",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "tools": ["*"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here" }
    }
  }
}
```

> **Important**: The JSON must be on a single line with no extra whitespace.

### 3. Using Environment Secrets (Recommended)

Instead of hardcoding tokens, use GitHub environment secrets:

1. Go to **Settings** → **Environments** → **copilot**
2. Add secret: `GITHUB_PERSONAL_ACCESS_TOKEN`
3. Update MCP config to reference the secret:

```json
{
  "mcpServers": {
    "github": {
      "type": "local",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "tools": ["*"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${{ secrets.GITHUB_PERSONAL_ACCESS_TOKEN }}" }
    }
  }
}
```

## Configuration Format Comparison

### Local QENEX LAB (`qenex.json`)

```json
{
  "mcp": {
    "<server-name>": {
      "type": "local",
      "command": ["<executable>", "<arg1>", "<arg2>"],
      "environment": {
        "KEY": "value"
      }
    }
  }
}
```

### GitHub Copilot (Single-line JSON)

```json
{
  "mcpServers": {
    "<server-name>": {
      "type": "local",
      "command": "<executable>",
      "args": ["<arg1>", "<arg2>"],
      "tools": ["*"],
      "env": { "KEY": "value" }
    }
  }
}
```

### Key Differences

| Feature        | Local Config            | GitHub Copilot Config                         |
| -------------- | ----------------------- | --------------------------------------------- |
| Root key       | `mcp`                   | `mcpServers`                                  |
| Command format | Array: `["cmd", "arg"]` | Separate: `"command": "cmd", "args": ["arg"]` |
| Environment    | `environment`           | `env`                                         |
| Tools filter   | Not needed              | Required: `"tools": ["*"]`                    |
| Format         | Pretty JSON             | Single-line JSON                              |

## Available MCP Servers

### Official Servers

| Server     | Package                                   | Purpose                         |
| ---------- | ----------------------------------------- | ------------------------------- |
| GitHub     | `@modelcontextprotocol/server-github`     | GitHub API (repos, issues, PRs) |
| Filesystem | `@modelcontextprotocol/server-filesystem` | File operations                 |
| Postgres   | `@modelcontextprotocol/server-postgres`   | Database queries                |
| Slack      | `@modelcontextprotocol/server-slack`      | Slack messaging                 |

### Custom MCP Server Template

QENEX LAB includes a custom MCP server template at `/mcp-server-template/`:

```bash
cd mcp-server-template
npm install
npm run build
npm start
```

Features:

- 7 example tools (api_query, search, create_item, etc.)
- 4 resources
- 4 prompt templates
- Configurable via environment variables

## Security Best Practices

1. **Never commit tokens** - Always use `.env` files (listed in `.gitignore`)
2. **Use environment secrets** - For GitHub Copilot, use GitHub environment secrets
3. **Rotate tokens immediately** if exposed in logs, commits, or conversations
4. **Minimal scopes** - Only grant required permissions to tokens
5. **Review MCP server code** - Audit third-party MCP servers before use

## Troubleshooting

### "Input must be valid JSON" on GitHub

- Ensure JSON is on a single line
- Remove all extra whitespace and newlines
- Validate JSON at jsonlint.com before pasting

### Token not found

- Check environment variable name matches exactly
- For local: Ensure `.env` file exists and is loaded
- For GitHub: Verify secret is in the `copilot` environment

### MCP server won't start

```bash
# Test locally
npx -y @modelcontextprotocol/server-github
```

Check that required environment variables are set.

## Files Reference

| File                   | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| `qenex.json`           | Local QENEX LAB MCP configuration        |
| `.env`                 | Local environment secrets (never commit) |
| `.env.example`         | Template for `.env` file                 |
| `mcp-server-template/` | Custom MCP server starter project        |

## Related Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- [GitHub Copilot Coding Agent Docs](https://docs.github.com/en/copilot/using-github-copilot/using-the-github-copilot-coding-agent)

---

_QENEX LAB v3.0-INFINITY - Air-gapped Scientific Computing Laboratory_

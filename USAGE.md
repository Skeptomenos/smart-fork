# Smart Fork Usage Guide for OpenCode

## Overview

Smart Fork integrates with OpenCode through:

1. **Custom Command** - `/detect-fork` for user invocation
2. **Agent Skill** - `detect-fork` for agent-initiated discovery
3. **CLI Tool** - `smart-fork` for ingestion and management

---

## Installation

### 1. Install the Python package

```bash
cd ~/.local/share/opencode/smart-fork
pip install -e .
```

### 2. Configure embedding provider

Edit `~/.local/share/opencode/smart-fork/config.json`:

```json
{
  "embedding": {
    "provider": "vertex",
    "vertex": {
      "project": "genaipilot-441014",
      "location": "us-central1",
      "model": "text-embedding-004"
    }
  }
}
```

### 3. Run initial sync

```bash
smart-fork sync
```

This indexes all existing sessions from `~/.local/share/opencode/sessions/`.

### 4. Install OpenCode integrations

Copy the command and skill to your global config:

```bash
# Custom command
mkdir -p ~/.config/opencode/commands
cp detect-fork.md ~/.config/opencode/commands/

# Agent skill
mkdir -p ~/.config/opencode/skills/detect-fork
cp SKILL.md ~/.config/opencode/skills/detect-fork/
```

---

## Usage

### Method 1: `/detect-fork` Command

The primary user interface. Type in any OpenCode session:

```
/detect-fork
```

OpenCode prompts you for intent, then searches and returns results.

**With inline arguments:**

```
/detect-fork implement webhook signature verification
```

**With scope:**

```
/detect-fork --scope repo continue the refactoring
```

### Method 2: Agent-Initiated

The agent can invoke the skill when it determines context would help:

```
You: I need to add rate limiting to this API

Claude: I notice this might benefit from prior context. Let me check
        for relevant sessions...
        
        [Invokes detect-fork skill]
        
        Found a highly relevant session from 3 days ago where we 
        implemented request throttling in gws-mcp-advanced. 
        
        Would you like to fork from that session?
        → opencode --session ses_abc123
```

### Method 3: CLI Direct

For scripting or quick lookups:

```bash
# Search all sessions
smart-fork search "webhook handling"

# Search within a repo
smart-fork search --repo /Users/david.helmus/repos/gws-mcp-advanced "oauth flow"

# Show stats
smart-fork status
```

---

## OpenCode Session Commands

OpenCode's native session handling:

| Command | Purpose |
|---------|---------|
| `opencode` | Start new session |
| `opencode --continue` or `-c` | Continue last session |
| `opencode --session <id>` or `-s <id>` | Resume specific session |
| `opencode session list` | List all sessions |

Smart Fork outputs the `--session` command ready to paste:

```
→ opencode --session ses_abc123def456
```

---

## File Structure

### OpenCode Integration Files

```
~/.config/opencode/
├── commands/
│   └── detect-fork.md          # /detect-fork command
└── skills/
    └── detect-fork/
        └── SKILL.md            # Agent skill definition
```

### Smart Fork Data

```
~/.local/share/opencode/smart-fork/
├── lance/
│   └── sessions.lance/         # Vector database
├── config.json                 # Provider & scoring config
├── sync-state.json             # Tracks last sync per repo
└── logs/
    └── sync.log
```

### OpenCode Sessions (source data)

```
~/.local/share/opencode/sessions/
├── ses_abc123/
│   ├── session.json            # Metadata
│   └── messages.json           # Transcript
├── ses_def456/
│   └── ...
└── ...
```

---

## Command Definition

`.config/opencode/commands/detect-fork.md`:

```markdown
---
description: Find the most relevant session to fork from
---

Find sessions relevant to: $ARGUMENTS

Search the Smart Fork vector database and return the top 5 matching 
sessions with relevance scores. For each result show:
- Score percentage
- Repository name  
- Time ago
- Brief context of what was discussed

Then provide the fork command for the top result:
→ opencode --session <session_id>
```

---

## Skill Definition

`.config/opencode/skills/detect-fork/SKILL.md`:

```markdown
---
name: detect-fork
description: Semantic search across all OpenCode sessions to find relevant prior context for forking
---

## Purpose

Search the Smart Fork RAG database to find previous sessions with 
context relevant to the current task. Use this when:

- Starting work that resembles past implementations
- User mentions something was "done before"
- Complex task that would benefit from prior context

## How to Use

1. Determine the user's intent (ask if unclear)
2. Call the smart-fork search API with the intent
3. Present ranked results with explanations
4. Offer the fork command for the best match

## Scoring Weights

Results are ranked by:
- Best chunk similarity (40%)
- Average similarity (20%)
- Recency (25%)
- Chunk coverage (5%)
- Chain quality (10%)

## Scopes

- **Global**: Search all sessions across all repos
- **Repo**: Filter to current repository only

## Output Format

Present results as a table:
| # | Score | Repo | When | Context |
|---|-------|------|------|---------|

Then provide: `→ opencode --session <id>`
```

---

## Workflow Examples

### Example 1: New Feature Implementation

```
$ opencode
You: /detect-fork add Stripe webhook handling

Claude: Found 5 relevant sessions:

┌─────┬───────┬─────────────────────┬──────────┬────────────────────────────────┐
│  #  │ Score │ Repo                │ When     │ Context                        │
├─────┼───────┼─────────────────────┼──────────┼────────────────────────────────┤
│  1  │  94%  │ account-management  │ 2 weeks  │ Stripe checkout & webhooks     │
│  2  │  88%  │ gws-mcp-advanced    │ 5 days   │ Webhook signature verification │
│  3  │  82%  │ its-fusion-kitchen  │ 1 month  │ Event-driven webhook dispatch  │
│  4  │  76%  │ drive-md            │ 2 months │ Payment processing flow        │
│  5  │  71%  │ loom                │ 3 months │ HTTP callback handling         │
└─────┴───────┴─────────────────────┴──────────┴────────────────────────────────┘

Top match implemented Stripe webhooks with signature verification.

→ opencode --session ses_stripe_abc123

Paste in a new terminal to fork with full context.
```

### Example 2: Repo-Scoped Continuation

```
You: /detect-fork --scope repo

Claude: What are you trying to accomplish?

You: Continue the API refactoring

Claude: Searching within current repo (gws-mcp-advanced)...

┌─────┬───────┬─────────────┬────────────────────────────────────────────┐
│  #  │ Score │ When        │ Context                                    │
├─────┼───────┼─────────────┼────────────────────────────────────────────┤
│  1  │  96%  │ Yesterday   │ Extracted handlers to service layer        │
│  2  │  89%  │ 2 days      │ Added TypeScript interfaces for API        │
│  3  │  84%  │ 4 days      │ Started router restructuring               │
└─────┴───────┴─────────────┴────────────────────────────────────────────┘

→ opencode --session ses_refactor_xyz789
```

### Example 3: Agent-Initiated Discovery

```
You: I need to implement OAuth PKCE flow for this CLI tool

Claude: This is a common pattern - let me check if you've implemented 
        this before...

        [Searches Smart Fork database]

        Found 2 relevant sessions:

        1. [91%] opencode - 1 month ago
           Implemented PKCE flow for OpenCode Zen authentication
        
        2. [78%] gws-mcp-advanced - 3 weeks ago  
           Google OAuth with refresh token handling

        Would you like to fork from the OpenCode session? It has the
        exact PKCE implementation pattern.

        → opencode --session ses_pkce_auth123

You: Yes, let's use that

Claude: Great! Open a new terminal and run:
        
        opencode --session ses_pkce_auth123
        
        That session has the full PKCE implementation including:
        - Code verifier generation
        - Authorization URL building
        - Token exchange logic
        - Secure storage of tokens
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `/detect-fork` not found | Check `~/.config/opencode/commands/detect-fork.md` exists |
| Skill not available | Verify `~/.config/opencode/skills/detect-fork/SKILL.md` |
| No results returned | Run `smart-fork sync` to index sessions |
| Stale results | Run `smart-fork sync --force` to re-index |
| Embedding errors | Check Vertex AI credentials or switch to Ollama |

---

## Background Sync (Optional)

For automatic session indexing, add a cron job:

```bash
# Every 30 minutes
*/30 * * * * ~/.local/share/opencode/smart-fork/bin/smart-fork sync --quiet
```

Or run the daemon:

```bash
smart-fork daemon &
```

# Smart Fork

Semantic session discovery for OpenCode. Find the most relevant sessions to fork from using natural language search.

## The Problem

Every OpenCode session accumulates valuable context—codebase understanding, solved problems, architectural decisions, debugging journeys. When a session ends, this context becomes trapped in transcript files, searchable only by filename or basic text grep.

With 1000+ sessions across 40+ repos, finding "that session where we implemented webhooks" is impossible.

## The Solution

Smart Fork indexes all your session transcripts into a vector database. Describe what you're trying to do, and it finds the sessions with the most relevant context:

```
$ smart-fork search "add webhook signature verification"

Found 5 relevant sessions:

 #  Score  Repo                  When      Context
 1   94%   gws-mcp-advanced      3 days    Implemented HMAC signature verification
 2   87%   account-management    1 week    Stripe webhook handling
 3   82%   its-fusion-kitchen    2 weeks   Event-driven webhook dispatch

→ opencode --session ses_abc123def456
```

Paste the fork command in a new terminal, and continue with full context.

## Installation

```bash
# Clone the repo
git clone https://github.com/Skeptomenos/smart-fork.git
cd smart-fork

# Install with uv
uv pip install -e .

# Run initial sync
smart-fork sync
```

## Usage

```bash
# Index all sessions
smart-fork sync

# Search for relevant sessions
smart-fork search "implement rate limiting"

# Search within current repo only
smart-fork search --scope repo "continue the refactoring"

# Show index stats
smart-fork status
```

## Automatic Sync (Daemon)

Smart Fork can monitor your sessions directory and automatically index new sessions in the background.

```bash
# Start background watcher
smart-fork daemon start

# Check status
smart-fork daemon status

# Enable auto-start on login (macOS)
smart-fork daemon enable
```

## OpenCode Integration

Smart Fork integrates with OpenCode via a custom command:

```
/detect-fork add webhook handling
```

See [USAGE.md](USAGE.md) for full integration details.

## Requirements

- Python 3.11+
- OpenCode with sessions at `~/.local/share/opencode/sessions/`
- Vertex AI API access (primary) or Ollama (fallback)

## Configuration

Edit `~/.local/share/opencode/smart-fork/config.json`:

```json
{
  "embedding": {
    "provider": "vertex",
    "vertex": {
      "project": "your-gcp-project",
      "location": "us-central1",
      "model": "text-embedding-004"
    }
  }
}
```

## Development

```bash
# Run tests
uv run pytest

# Type check
uv run mypy --strict smart_fork/

# Run CLI
uv run smart-fork --help
```

## Documentation

- [VISION.md](VISION.md) - Problem statement and solution narrative
- [USAGE.md](USAGE.md) - Detailed usage guide with examples
- [ralph-wiggum/prds/PRD.md](ralph-wiggum/prds/PRD.md) - Product requirements
- [ralph-wiggum/specs/spec-smart-fork.md](ralph-wiggum/specs/spec-smart-fork.md) - Technical specification

## Status

**Complete** - All core features implemented.

- [x] Semantic search with Vertex AI / Ollama
- [x] LanceDB vector storage
- [x] OpenCode integration (command + skill)
- [x] Background sync daemon
- [x] MCP server

See [ralph-wiggum/code/plan.md](session-reference/ralph-wiggum/code/plan.md) for implementation details.

## License

MIT

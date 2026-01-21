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
git clone https://github.com/dhelms-bw/smart-fork.git
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

**In Development** - Phase 1 (Foundation) in progress.

See [ralph-wiggum/code/plan.md](ralph-wiggum/code/plan.md) for implementation progress.

## License

MIT

# Smart Fork

Semantic session discovery for OpenCode. Indexes session transcripts into a vector database and enables natural language search to find the most relevant sessions for forking.

## Stack

- **Language:** Python 3.11+
- **Package Manager:** uv
- **Vector DB:** LanceDB (embedded, no server)
- **Embeddings:** Vertex AI `text-embedding-004` (primary), Ollama `nomic-embed-text` (fallback)
- **Token Counting:** tiktoken

## Project Structure

```
smart_fork/
├── cli.py           # CLI entry point (smart-fork command)
├── ingest.py        # Session ingestion pipeline
├── query.py         # Search and scoring logic
├── embeddings.py    # Vertex AI / Ollama provider abstraction
├── config.py        # Configuration loading
└── types.py         # Data classes and schemas

tests/               # pytest tests
```

## Key Paths

| Path | Purpose |
|------|---------|
| `~/.local/share/opencode/sessions/` | Source session transcripts (read-only) |
| `~/.local/share/opencode/smart-fork/lance/` | LanceDB vector storage |
| `~/.local/share/opencode/smart-fork/config.json` | User configuration |

## Commands

```bash
uv run smart-fork sync          # Index all sessions
uv run smart-fork search "..."  # Query sessions
uv run smart-fork status        # Show index stats
uv run pytest                   # Run tests
uv run mypy --strict            # Type check
```

## Conventions

- Type hints required on all public functions
- Use `pathlib.Path` for file operations
- Logging via `structlog`
- Config via pydantic models

## Documentation

| Document | Purpose |
|----------|---------|
| `ralph-wiggum/prds/PRD.md` | Product requirements with acceptance criteria |
| `ralph-wiggum/specs/spec-smart-fork.md` | Technical specification and data models |
| `ralph-wiggum/code/plan.md` | Implementation plan with task tracking |
| `USAGE.md` | User-facing usage guide |

## Current Status

Development in progress. Code lives in `smart_fork/` (this repo).
Runtime data will be stored in `~/.local/share/opencode/smart-fork/` when deployed.

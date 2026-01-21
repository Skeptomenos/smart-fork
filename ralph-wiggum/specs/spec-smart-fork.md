# Smart Fork - Technical Specification

> **Status**: Approved by Architect
> **Type**: Feature (Greenfield)
> **PRD**: @ralph-wiggum/prds/PRD.md
> **Created**: 2025-01-21

---

## Context

Smart Fork enables semantic discovery of OpenCode sessions using vector search. Users describe their intent in natural language, and the system finds the most relevant prior sessions to fork from, reducing context re-establishment from 3-5 messages to zero.

**Core value**: Transform 1000+ session transcripts from forgotten artifacts into a searchable knowledge base.

---

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Smart Fork CLI                            │
│  smart-fork sync | search | status                                  │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│                         Core Library                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Ingestion  │  │    Query     │  │   Scoring    │              │
│  │   Pipeline   │  │   Engine     │  │   Engine     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘              │
│         │                 │                                         │
│  ┌──────▼─────────────────▼──────┐  ┌──────────────┐              │
│  │         Embedding Provider     │  │   Chunker    │              │
│  │  (Vertex AI / Ollama)         │  │              │              │
│  └────────────────────────────────┘  └──────────────┘              │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│                         LanceDB                                      │
│  ~/.local/share/opencode/smart-fork/lance/sessions.lance            │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Model

```python
# src/smart_fork/models.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class SessionChunk:
    """A single indexed chunk from a session transcript."""
    id: str                    # "{session_id}_chunk_{index}"
    session_id: str            # e.g., "ses_abc123def456"
    repo_path: str             # Absolute path to repo root
    chunk_index: int           # Position within session (0-based)
    chunk_text: str            # Raw chunk content
    embedding: list[float]     # Vector (768 dimensions)
    timestamp: int             # Unix timestamp of session
    model_used: str            # Embedding model identifier
    token_count: int           # Tokens in chunk

@dataclass
class SessionMatch:
    """Query result representing a matched session."""
    session_id: str
    repo_path: str
    repo_name: str             # basename(repo_path)
    timestamp: int
    score: float               # Composite score 0.0-1.0
    best_snippet: str          # Truncated best-matching chunk
    chunk_count: int           # Number of matching chunks

@dataclass
class SyncState:
    """Tracks ingestion state for incremental sync."""
    last_sync: int             # Unix timestamp
    sessions: dict[str, int]   # session_id -> last_modified_timestamp
```

### Embedding Providers

| Provider | Model | Dimensions | Config Key |
|----------|-------|------------|------------|
| Vertex AI | `text-embedding-004` | 768 | `vertex` |
| Ollama | `nomic-embed-text` | 768 | `ollama` |

Provider interface:
```python
# src/smart_fork/embedding/base.py
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...
    
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for storage."""
        ...
```

### Scoring Algorithm

```python
def compute_session_score(
    chunks: list[ChunkMatch],
    total_chunks_in_session: int,
    session_timestamp: int,
    query_timestamp: int,
    parent_session_id: Optional[str],
) -> float:
    """
    Weights:
    - best_similarity: 0.40 (highest matching chunk)
    - avg_similarity:  0.20 (mean of all matching chunks)
    - chunk_ratio:     0.05 (matching chunks / total chunks)
    - recency:         0.25 (exponential decay, 30-day half-life)
    - chain_quality:   0.10 (bonus if session has parent link)
    """
```

### File Structure

**Package (installable via pip -e):**
```
~/.local/share/opencode/smart-fork/
├── src/
│   └── smart_fork/
│       ├── __init__.py
│       ├── cli.py              # Click CLI entrypoint
│       ├── models.py           # Dataclasses
│       ├── config.py           # Config loading
│       ├── chunker.py          # Token-based chunking
│       ├── db.py               # LanceDB wrapper
│       ├── ingestion.py        # Sync pipeline
│       ├── query.py            # Search logic
│       ├── scoring.py          # Composite scoring
│       └── embedding/
│           ├── __init__.py
│           ├── base.py         # Abstract provider
│           ├── vertex.py       # Vertex AI provider
│           └── ollama.py       # Ollama provider
├── tests/
│   ├── __init__.py
│   ├── test_chunker.py
│   ├── test_ingestion.py
│   ├── test_query.py
│   └── test_scoring.py
├── pyproject.toml
├── lance/                      # LanceDB data (created at runtime)
│   └── sessions.lance/
├── config.json
├── sync-state.json
└── logs/
    └── sync.log
```

**OpenCode integrations:**
```
~/.config/opencode/
├── commands/
│   └── detect-fork.md
└── skills/
    └── detect-fork/
        └── SKILL.md
```

---

## Requirements Summary

From PRD (must-have stories):
1. **US-001**: Ingest sessions to LanceDB with Vertex AI embeddings
2. **US-002**: Query by natural language, return top 5 sessions
3. **US-003**: Global and repo-scoped search
4. **US-004**: Output `opencode --session <id>` command
5. **US-005**: `/detect-fork` OpenCode command
6. **US-006**: `detect-fork` agent skill

Should-have:
7. **US-007**: Ollama fallback for offline use
8. **US-008**: Incremental sync

---

## Implementation Plan (Atomic Tasks)

### Phase 1: Project Setup
- [ ] 1.1 Initialize Python package structure
      Files: `pyproject.toml`, `src/smart_fork/__init__.py`
      Test: `pip install -e .` succeeds
- [ ] 1.2 Create config module with schema validation
      Files: `src/smart_fork/config.py`
      Test: Loads config.json, validates required keys
- [ ] 1.3 Create data models
      Files: `src/smart_fork/models.py`
      Test: `mypy --strict` passes

### Phase 2: Chunking
- [ ] 2.1 Implement token-based chunker
      Files: `src/smart_fork/chunker.py`
      Test: Splits text into 512-1024 token chunks with 50-token overlap
      Notes: Use `tiktoken` with `cl100k_base` encoding
- [ ] 2.2 Add session transcript parser
      Files: `src/smart_fork/chunker.py`
      Test: Reads `messages.json` from session dir, extracts text
- [ ] 2.3 Write chunker unit tests
      Files: `tests/test_chunker.py`
      Test: `pytest tests/test_chunker.py` passes

### Phase 3: Embedding Providers
- [ ] 3.1 Create abstract EmbeddingProvider base class
      Files: `src/smart_fork/embedding/base.py`
      Test: `mypy --strict` passes
- [ ] 3.2 Implement Vertex AI provider
      Files: `src/smart_fork/embedding/vertex.py`
      Test: Embeds test string, returns 768-dim vector
      Notes: Use `google-cloud-aiplatform` SDK, batch requests
- [ ] 3.3 Implement Ollama provider
      Files: `src/smart_fork/embedding/ollama.py`
      Test: Embeds test string via `http://localhost:11434/api/embeddings`
- [ ] 3.4 Create provider factory
      Files: `src/smart_fork/embedding/__init__.py`
      Test: `get_provider("vertex")` returns VertexProvider

### Phase 4: LanceDB Integration
- [ ] 4.1 Create LanceDB wrapper
      Files: `src/smart_fork/db.py`
      Test: Creates table if not exists, adds chunks
- [ ] 4.2 Define schema for sessions.lance table
      Files: `src/smart_fork/db.py`
      Schema: id, session_id, repo_path, chunk_index, chunk_text, embedding (768-dim), timestamp, model_used, token_count
      Test: Schema validation passes
- [ ] 4.3 Implement vector search with filters
      Files: `src/smart_fork/db.py`
      Test: Query returns top-k results filtered by repo_path

### Phase 5: Ingestion Pipeline (US-001)
- [ ] 5.1 Implement session discovery
      Files: `src/smart_fork/ingestion.py`
      Test: Finds all `ses_*` directories in `~/.local/share/opencode/sessions/`
- [ ] 5.2 Extract repo_path from session metadata
      Files: `src/smart_fork/ingestion.py`
      Test: Parses `session.json`, extracts working directory
- [ ] 5.3 Implement full sync pipeline
      Files: `src/smart_fork/ingestion.py`
      Test: Ingests 10 test sessions, chunks stored in LanceDB
- [ ] 5.4 Add sync state tracking
      Files: `src/smart_fork/ingestion.py`
      Test: Updates `sync-state.json` after successful sync
- [ ] 5.5 Implement incremental sync (US-008)
      Files: `src/smart_fork/ingestion.py`
      Test: Only processes sessions modified after last sync
- [ ] 5.6 Add `--force` flag for full re-index
      Files: `src/smart_fork/ingestion.py`, `src/smart_fork/cli.py`
      Test: `smart-fork sync --force` re-indexes all
- [ ] 5.7 Add rate limiting (100ms delay between API calls)
      Files: `src/smart_fork/embedding/vertex.py`
      Test: Observes delay between batch requests
- [ ] 5.8 Write ingestion tests
      Files: `tests/test_ingestion.py`
      Test: `pytest tests/test_ingestion.py` passes

### Phase 6: Query Engine (US-002, US-003)
- [ ] 6.1 Implement query embedding
      Files: `src/smart_fork/query.py`
      Test: Embeds query string, returns vector
- [ ] 6.2 Implement chunk search (top 20)
      Files: `src/smart_fork/query.py`
      Test: Searches LanceDB, returns matching chunks
- [ ] 6.3 Implement repo scope filter (US-003)
      Files: `src/smart_fork/query.py`
      Test: `--scope repo` filters by current directory
- [ ] 6.4 Group chunks by session
      Files: `src/smart_fork/query.py`
      Test: Chunks from same session aggregated
- [ ] 6.5 Write query tests
      Files: `tests/test_query.py`
      Test: `pytest tests/test_query.py` passes

### Phase 7: Scoring Engine
- [ ] 7.1 Implement composite scoring algorithm
      Files: `src/smart_fork/scoring.py`
      Test: Score computed with correct weights
- [ ] 7.2 Implement recency decay (30-day half-life)
      Files: `src/smart_fork/scoring.py`
      Test: 30-day-old session has 50% recency score
- [ ] 7.3 Implement chain quality bonus
      Files: `src/smart_fork/scoring.py`
      Test: Sessions with parent_session_id get bonus
- [ ] 7.4 Write scoring tests
      Files: `tests/test_scoring.py`
      Test: `pytest tests/test_scoring.py` passes

### Phase 8: CLI (US-004)
- [ ] 8.1 Create Click CLI entrypoint
      Files: `src/smart_fork/cli.py`
      Test: `smart-fork --help` shows commands
- [ ] 8.2 Implement `smart-fork sync` command
      Files: `src/smart_fork/cli.py`
      Test: Runs ingestion pipeline
- [ ] 8.3 Implement `smart-fork search <query>` command
      Files: `src/smart_fork/cli.py`
      Test: Returns formatted table of results
- [ ] 8.4 Implement `smart-fork status` command
      Files: `src/smart_fork/cli.py`
      Test: Shows indexed session count, last sync time
- [ ] 8.5 Add `--scope` and `--repo` flags
      Files: `src/smart_fork/cli.py`
      Test: `smart-fork search --scope repo "query"` filters results
- [ ] 8.6 Output fork command for top result (US-004)
      Files: `src/smart_fork/cli.py`
      Test: Outputs `opencode --session <id>` at end
- [ ] 8.7 Handle no-results gracefully
      Files: `src/smart_fork/cli.py`
      Test: Shows helpful message with tips

### Phase 9: OpenCode Integrations (US-005, US-006)
- [ ] 9.1 Create `/detect-fork` command
      Files: `~/.config/opencode/commands/detect-fork.md`
      Test: `/detect-fork` works in OpenCode TUI
      Content:
      ```markdown
      ---
      description: Find the most relevant session to fork from
      ---
      
      Find sessions relevant to: $ARGUMENTS
      
      Run `smart-fork search "$ARGUMENTS"` and format results as table.
      If no arguments, ask user what they're trying to accomplish.
      Show fork command for top result.
      ```
- [ ] 9.2 Create `detect-fork` agent skill
      Files: `~/.config/opencode/skills/detect-fork/SKILL.md`
      Test: Agent can invoke skill proactively
      Content: See USAGE.md for full skill definition

### Phase 10: Quality Assurance
- [ ] 10.1 Run mypy --strict on all modules
      Test: Zero type errors
- [ ] 10.2 Run full test suite
      Test: `pytest` passes with >80% coverage
- [ ] 10.3 Performance test: ingest 100 sessions
      Test: Completes in < 10 minutes
- [ ] 10.4 Performance test: query latency
      Test: Search completes in < 3 seconds
- [ ] 10.5 Manual verification: run 10 semantic queries
      Test: Relevant result in top 5 for >90% of queries

---

## Verification Checklist

After implementation, verify:

- [ ] `pip install -e ~/.local/share/opencode/smart-fork` succeeds
- [ ] `smart-fork sync` indexes sessions without errors
- [ ] `smart-fork search "webhook handling"` returns relevant results
- [ ] `smart-fork search --scope repo "continue refactoring"` filters correctly
- [ ] `smart-fork status` shows session count and sync time
- [ ] `/detect-fork` command works in OpenCode
- [ ] Fork command `opencode --session <id>` actually resumes the session
- [ ] `mypy --strict src/` passes
- [ ] `pytest tests/` passes

---

## Dependencies

**Python packages (add to pyproject.toml):**
```toml
[project]
dependencies = [
    "click>=8.0",
    "lancedb>=0.4",
    "tiktoken>=0.5",
    "google-cloud-aiplatform>=1.40",
    "httpx>=0.25",          # For Ollama
    "rich>=13.0",           # For table output
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.5",
]
```

---

## Open Decisions (From PRD)

These are documented in PRD Section 14, agent should use PRD's "leaning" guidance:

1. **Session retention**: Keep forever (no aging out in v1)
2. **Multi-repo sessions**: Use primary working directory
3. **Index content**: Both user and assistant messages
4. **Chunk boundaries**: Token count with message awareness
5. **Chain quality**: Parent link only (no fork depth)

---

## Constraints Reminder

From PRD Section 7 (AI Agent Boundaries):

**Agent CAN decide**: Naming, internal organization, utility functions, test structure

**Agent MUST ASK before**: New dependencies, schema changes, scoring weight changes

**Agent CANNOT**: Skip acceptance criteria, modify OpenCode's session format, store API keys in DB

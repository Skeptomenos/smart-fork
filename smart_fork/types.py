"""Core data models for Smart Fork.

These dataclasses define the schema for session chunks stored in LanceDB,
query results returned to users, and sync state for incremental updates.

Why dataclasses over pydantic for core types:
- LanceDB works directly with dataclasses
- Simpler serialization to Arrow format
- Pydantic is used for config validation (config.py), not storage models
"""

from dataclasses import dataclass, field


@dataclass
class SessionChunk:
    """A single indexed chunk from a session transcript.

    Each session transcript is split into multiple chunks for embedding.
    Chunks are the unit of storage in LanceDB and the unit of vector search.

    Attributes:
        id: Unique identifier in format "{session_id}_chunk_{chunk_index}"
        session_id: OpenCode session ID (e.g., "ses_abc123def456")
        repo_path: Absolute path to the repository root where session occurred
        chunk_index: Zero-based position of this chunk within the session
        chunk_text: Raw text content of this chunk
        embedding: 768-dimensional vector from embedding model
        timestamp: Unix timestamp of the session creation
        model_used: Identifier of the embedding model (e.g., "text-embedding-004")
        token_count: Number of tokens in chunk_text (for debugging/analysis)
    """

    id: str
    session_id: str
    repo_path: str
    chunk_index: int
    chunk_text: str
    embedding: list[float]
    timestamp: int
    model_used: str
    token_count: int


@dataclass
class ChunkMatch:
    """A single chunk returned from vector search.

    Represents a matched chunk with its similarity score. Used as input
    to the scoring algorithm which aggregates chunks into SessionMatch.

    Attributes:
        chunk: The matched SessionChunk with full data
        similarity: Similarity score from 0.0 to 1.0 (higher = more similar)
                   Converted from LanceDB distance (lower = closer)
        distance: Raw distance from LanceDB (for debugging/analysis)
    """

    chunk: SessionChunk
    similarity: float
    distance: float


@dataclass
class SessionMatch:
    """Query result representing a matched session.

    Returned by the query engine after searching for relevant sessions.
    Groups multiple matching chunks into a single session result with
    a composite score.

    Attributes:
        session_id: OpenCode session ID for forking
        repo_path: Absolute path to the repository root
        repo_name: basename(repo_path) for display purposes
        timestamp: Unix timestamp of the session creation
        score: Composite relevance score from 0.0 to 1.0
        best_snippet: Truncated text from the highest-scoring chunk
        chunk_count: Number of chunks from this session that matched the query
    """

    session_id: str
    repo_path: str
    repo_name: str
    timestamp: int
    score: float
    best_snippet: str
    chunk_count: int


@dataclass
class SyncState:
    """Tracks ingestion state for incremental sync.

    Persisted to ~/.local/share/opencode/smart-fork/sync-state.json
    to enable efficient incremental updates. Only sessions modified
    after last_sync need to be re-indexed.

    Attributes:
        last_sync: Unix timestamp of the last successful sync completion
        sessions: Map of session_id to last_modified_timestamp for each
                  indexed session. Used to detect modified sessions.
    """

    last_sync: int
    sessions: dict[str, int] = field(default_factory=dict)

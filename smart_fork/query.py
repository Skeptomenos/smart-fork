"""Query engine for Smart Fork.

Implements semantic search over indexed session chunks:
1. Embed the query text using the configured provider
2. Search LanceDB for similar chunks (top-k ANN)
3. Group chunks by session
4. Return matches ordered by similarity

Task 5.1 implements the core query→embed→search→return flow.
Tasks 5.2-5.3 will add composite scoring and result formatting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from smart_fork.config import SmartForkConfig, load_config
from smart_fork.db import ChunkDatabase
from smart_fork.embeddings import EmbeddingProvider, create_provider
from smart_fork.types import ChunkMatch, SessionMatch


# --- Exceptions ---


class QueryError(Exception):
    """Base exception for query engine errors."""

    pass


class EmptyQueryError(QueryError):
    """Raised when query text is empty or whitespace-only."""

    pass


class NoResultsError(QueryError):
    """Raised when no matching sessions are found.

    Not necessarily an error - may just mean no relevant content indexed.
    """

    pass


# --- Query Result Types ---


@dataclass
class QueryResult:
    """Result of a search_sessions() call.

    Attributes:
        matches: List of SessionMatch objects, ordered by similarity (descending).
        query_time_ms: Time taken for the query in milliseconds.
        total_chunks_searched: Number of chunks examined in vector search.
    """

    matches: list[SessionMatch]
    query_time_ms: float
    total_chunks_searched: int


@dataclass
class SessionChunkGroup:
    """Internal grouping of chunks belonging to a single session.

    Used for aggregating chunk matches before computing session scores.
    """

    session_id: str
    repo_path: str
    timestamp: int
    chunks: list[ChunkMatch]

    @property
    def best_similarity(self) -> float:
        """Highest similarity score among chunks."""
        return max(c.similarity for c in self.chunks) if self.chunks else 0.0

    @property
    def avg_similarity(self) -> float:
        """Mean similarity score across chunks."""
        if not self.chunks:
            return 0.0
        return sum(c.similarity for c in self.chunks) / len(self.chunks)

    @property
    def best_chunk(self) -> ChunkMatch | None:
        """Chunk with highest similarity score."""
        if not self.chunks:
            return None
        return max(self.chunks, key=lambda c: c.similarity)


# --- Core Query Functions ---


def search_sessions(
    query: str,
    *,
    config: SmartForkConfig | None = None,
    db: ChunkDatabase | None = None,
    provider: EmbeddingProvider | None = None,
    repo_path: str | None = None,
    top_k: int | None = None,
    top_results: int | None = None,
) -> QueryResult:
    """Search for sessions semantically similar to the query.

    This is the main entry point for querying the session index.

    Args:
        query: Natural language query describing the desired session context.
        config: Configuration to use. If None, loads from default location.
        db: Database instance to search. If None, opens from config paths.
        provider: Embedding provider. If None, creates from config.
        repo_path: If provided, filter results to this repository only.
        top_k: Override for number of chunks to retrieve (default from config).
        top_results: Override for number of sessions to return (default from config).

    Returns:
        QueryResult with matches ordered by similarity (descending).

    Raises:
        EmptyQueryError: If query is empty or whitespace-only.
        QueryError: If embedding or search fails.

    Example:
        >>> result = search_sessions("implement rate limiting for API calls")
        >>> for match in result.matches:
        ...     print(f"{match.session_id}: {match.score:.2f}")
    """
    start_time = time.perf_counter()

    # Validate query
    query = query.strip()
    if not query:
        raise EmptyQueryError("Query cannot be empty")

    # Load config and create dependencies if not provided
    if config is None:
        config = load_config()

    if db is None:
        db = ChunkDatabase.open(config.paths.lance_path)

    if provider is None:
        provider = create_provider(config.embedding)

    # Apply defaults from config
    if top_k is None:
        top_k = config.query.top_chunks
    if top_results is None:
        top_results = config.query.top_results

    # Embed the query text
    try:
        embeddings = provider.embed([query])
        query_vector = embeddings[0]
    except Exception as e:
        raise QueryError(f"Failed to embed query: {e}") from e

    # Search for similar chunks
    try:
        chunk_matches = db.search(
            query_vector,
            top_k=top_k,
            repo_path=repo_path,
        )
    except Exception as e:
        raise QueryError(f"Failed to search database: {e}") from e

    # Group chunks by session
    session_groups = _group_chunks_by_session(chunk_matches)

    # Convert to SessionMatch objects (simple similarity-based ranking for Task 5.1)
    # Task 5.2 will add composite scoring
    matches = _create_session_matches(session_groups, top_results)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return QueryResult(
        matches=matches,
        query_time_ms=elapsed_ms,
        total_chunks_searched=len(chunk_matches),
    )


def _group_chunks_by_session(
    chunk_matches: list[ChunkMatch],
) -> dict[str, SessionChunkGroup]:
    """Group chunk matches by their session ID.

    Args:
        chunk_matches: List of ChunkMatch objects from vector search.

    Returns:
        Dictionary mapping session_id to SessionChunkGroup.
    """
    groups: dict[str, SessionChunkGroup] = {}

    for match in chunk_matches:
        session_id = match.chunk.session_id

        if session_id not in groups:
            groups[session_id] = SessionChunkGroup(
                session_id=session_id,
                repo_path=match.chunk.repo_path,
                timestamp=match.chunk.timestamp,
                chunks=[],
            )

        groups[session_id].chunks.append(match)

    return groups


def _create_session_matches(
    groups: dict[str, SessionChunkGroup],
    top_results: int,
) -> list[SessionMatch]:
    """Convert session groups to SessionMatch objects.

    For Task 5.1, uses simple best_similarity ranking.
    Task 5.2 will implement full composite scoring.

    Args:
        groups: Dictionary of session groups from _group_chunks_by_session.
        top_results: Maximum number of results to return.

    Returns:
        List of SessionMatch objects, sorted by score (descending).
    """
    matches: list[SessionMatch] = []

    for group in groups.values():
        best_chunk = group.best_chunk
        if best_chunk is None:
            continue

        # Extract snippet from best matching chunk (truncate to ~100 chars)
        snippet = best_chunk.chunk.chunk_text[:200]
        if len(best_chunk.chunk.chunk_text) > 200:
            snippet = snippet.rsplit(" ", 1)[0] + "..."

        # For Task 5.1: Use best_similarity as score
        # Task 5.2 will implement full composite scoring algorithm
        score = group.best_similarity

        match = SessionMatch(
            session_id=group.session_id,
            repo_path=group.repo_path,
            repo_name=Path(group.repo_path).name,
            timestamp=group.timestamp,
            score=score,
            best_snippet=snippet,
            chunk_count=len(group.chunks),
        )
        matches.append(match)

    # Sort by score descending
    matches.sort(key=lambda m: m.score, reverse=True)

    # Return top N results
    return matches[:top_results]


def embed_query(
    query: str,
    *,
    config: SmartForkConfig | None = None,
    provider: EmbeddingProvider | None = None,
) -> list[float]:
    """Embed a query string into a vector.

    This is a convenience function for getting just the embedding.
    Most callers should use search_sessions() instead.

    Args:
        query: The query text to embed.
        config: Configuration to use. If None, loads from default location.
        provider: Embedding provider. If None, creates from config.

    Returns:
        768-dimensional embedding vector.

    Raises:
        EmptyQueryError: If query is empty or whitespace-only.
        QueryError: If embedding fails.
    """
    query = query.strip()
    if not query:
        raise EmptyQueryError("Query cannot be empty")

    if config is None:
        config = load_config()

    if provider is None:
        provider = create_provider(config.embedding)

    try:
        embeddings = provider.embed([query])
        return embeddings[0]
    except Exception as e:
        raise QueryError(f"Failed to embed query: {e}") from e

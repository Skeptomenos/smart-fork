"""Query engine for Smart Fork.

Implements semantic search over indexed session chunks:
1. Embed the query text using the configured provider
2. Search LanceDB for similar chunks (top-k ANN)
3. Group chunks by session
4. Compute composite scores using weighted algorithm
5. Return matches ordered by score

The composite scoring algorithm (compute_session_score) combines:
- best_similarity (40%): Highest matching chunk similarity
- avg_similarity (20%): Mean similarity across matching chunks
- chunk_ratio (5%): Matching chunks / total chunks in session
- recency (25%): Exponential decay with 30-day half-life
- chain_quality (10%): Bonus if session was forked from parent
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from smart_fork.config import (
    QueryConfig,
    ScoringWeights,
    SmartForkConfig,
    load_config,
)
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


# --- Scoring Functions ---


def compute_session_score(
    *,
    best_similarity: float,
    avg_similarity: float,
    matching_chunks: int,
    total_chunks: int,
    session_timestamp: int,
    query_timestamp: int,
    has_parent: bool,
    weights: ScoringWeights,
    half_life_days: int = 30,
) -> float:
    """Compute composite relevance score for a session.

    Combines multiple signals into a single 0.0-1.0 score using configurable
    weights. Each component is normalized to 0.0-1.0 before weighting.

    Args:
        best_similarity: Highest similarity score among matching chunks (0.0-1.0).
        avg_similarity: Mean similarity across matching chunks (0.0-1.0).
        matching_chunks: Number of chunks from this session that matched.
        total_chunks: Total chunks in the session (for ratio calculation).
        session_timestamp: Unix timestamp when session was created.
        query_timestamp: Unix timestamp when query was issued.
        has_parent: True if session has a parent link (was forked).
        weights: ScoringWeights configuration with component weights.
        half_life_days: Number of days for recency half-life decay.

    Returns:
        Composite score from 0.0 to 1.0 (higher = more relevant).

    Example:
        >>> score = compute_session_score(
        ...     best_similarity=0.85,
        ...     avg_similarity=0.72,
        ...     matching_chunks=3,
        ...     total_chunks=10,
        ...     session_timestamp=1705000000,
        ...     query_timestamp=1706000000,
        ...     has_parent=True,
        ...     weights=ScoringWeights(),
        ... )
    """
    # Component 1: Best similarity (already 0-1)
    best_sim_component = best_similarity * weights.best_similarity

    # Component 2: Average similarity (already 0-1)
    avg_sim_component = avg_similarity * weights.avg_similarity

    # Component 3: Chunk ratio (matching / total, 0-1)
    # Avoid division by zero
    if total_chunks > 0:
        chunk_ratio = min(matching_chunks / total_chunks, 1.0)
    else:
        chunk_ratio = 0.0
    chunk_ratio_component = chunk_ratio * weights.chunk_ratio

    # Component 4: Recency (exponential decay with half-life)
    # recency = 0.5 ^ (age_days / half_life_days)
    # More recent = higher score, approaches 0 as age increases
    age_seconds = max(0, query_timestamp - session_timestamp)
    age_days = age_seconds / (24 * 60 * 60)
    recency = math.pow(0.5, age_days / half_life_days)
    recency_component = recency * weights.recency

    # Component 5: Chain quality (binary: 1.0 if has parent, 0.0 otherwise)
    chain_quality = 1.0 if has_parent else 0.0
    chain_component = chain_quality * weights.chain_quality

    # Sum all weighted components
    total_score = (
        best_sim_component
        + avg_sim_component
        + chunk_ratio_component
        + recency_component
        + chain_component
    )

    # Clamp to 0-1 range (should already be in range if weights sum to 1)
    return max(0.0, min(1.0, total_score))


def compute_recency_score(
    session_timestamp: int,
    query_timestamp: int,
    half_life_days: int = 30,
) -> float:
    """Compute recency score using exponential decay.

    Uses the formula: recency = 0.5 ^ (age_days / half_life_days)

    Args:
        session_timestamp: Unix timestamp when session was created.
        query_timestamp: Unix timestamp when query was issued.
        half_life_days: Number of days for the half-life decay.

    Returns:
        Recency score from 0.0 to 1.0 (1.0 = just now, 0.5 = half_life_days ago).
    """
    age_seconds = max(0, query_timestamp - session_timestamp)
    age_days = age_seconds / (24 * 60 * 60)
    return math.pow(0.5, age_days / half_life_days)


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

    # Get current timestamp for recency scoring
    query_timestamp = int(time.time())

    # Convert to SessionMatch objects with composite scoring
    matches = _create_session_matches(
        groups=session_groups,
        top_results=top_results,
        db=db,
        config=config,
        query_timestamp=query_timestamp,
        sessions_dir=config.paths.sessions_dir,
    )

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
    *,
    db: ChunkDatabase,
    config: SmartForkConfig,
    query_timestamp: int,
    sessions_dir: Path,
) -> list[SessionMatch]:
    """Convert session groups to SessionMatch objects with composite scoring.

    Uses the composite scoring algorithm from compute_session_score() which
    combines similarity, recency, chunk ratio, and chain quality signals.

    Args:
        groups: Dictionary of session groups from _group_chunks_by_session.
        top_results: Maximum number of results to return.
        db: Database for looking up total chunk counts per session.
        config: Configuration with scoring weights and recency half-life.
        query_timestamp: Unix timestamp when query was issued (for recency).
        sessions_dir: Path to sessions directory for parent lookup.

    Returns:
        List of SessionMatch objects, sorted by composite score (descending).
    """
    matches: list[SessionMatch] = []

    # Cache parent lookups to avoid repeated file reads
    parent_cache: dict[str, bool] = {}

    for group in groups.values():
        best_chunk = group.best_chunk
        if best_chunk is None:
            continue

        # Extract snippet from best matching chunk (truncate to ~200 chars)
        snippet = best_chunk.chunk.chunk_text[:200]
        if len(best_chunk.chunk.chunk_text) > 200:
            snippet = snippet.rsplit(" ", 1)[0] + "..."

        # Get total chunks for this session (for chunk_ratio calculation)
        total_chunks = db.get_session_chunk_count(group.session_id)
        if total_chunks == 0:
            # Fallback if count not available
            total_chunks = len(group.chunks)

        # Check if session has a parent (for chain_quality)
        has_parent = _get_session_has_parent(
            group.session_id, sessions_dir, parent_cache
        )

        # Compute composite score
        score = compute_session_score(
            best_similarity=group.best_similarity,
            avg_similarity=group.avg_similarity,
            matching_chunks=len(group.chunks),
            total_chunks=total_chunks,
            session_timestamp=group.timestamp,
            query_timestamp=query_timestamp,
            has_parent=has_parent,
            weights=config.scoring,
            half_life_days=config.query.recency_half_life_days,
        )

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


def _get_session_has_parent(
    session_id: str,
    sessions_dir: Path,
    cache: dict[str, bool],
) -> bool:
    """Check if a session has a parent link (was forked).

    Reads session.json to check for parent_session_id or forked_from field.
    Results are cached to avoid repeated file I/O for the same session.

    Args:
        session_id: The session ID to check.
        sessions_dir: Path to the sessions directory.
        cache: Dictionary for caching results.

    Returns:
        True if session has a parent, False otherwise (including on errors).
    """
    if session_id in cache:
        return cache[session_id]

    has_parent = False
    session_path = sessions_dir / session_id / "session.json"

    try:
        if session_path.exists():
            with open(session_path) as f:
                data = json.load(f)
            # Check both field names for compatibility
            parent = data.get("parent_session_id") or data.get("forked_from")
            has_parent = bool(parent)
    except (OSError, json.JSONDecodeError):
        # On error, assume no parent
        pass

    cache[session_id] = has_parent
    return has_parent


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

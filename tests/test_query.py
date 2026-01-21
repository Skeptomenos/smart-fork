"""Tests for the query engine module.

Tests cover:
- Query embedding and validation
- Session search with vector matching
- Session grouping and result formatting
- Repo-scoped filtering
- Error handling
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from smart_fork.config import (
    EmbeddingConfig,
    PathsConfig,
    QueryConfig,
    SmartForkConfig,
    ScoringWeights,
)
from smart_fork.db import ChunkDatabase, VECTOR_DIMENSIONS
from smart_fork.embeddings import EmbeddingProvider
from smart_fork.query import (
    EmptyQueryError,
    QueryError,
    QueryResult,
    SessionChunkGroup,
    embed_query,
    search_sessions,
    _group_chunks_by_session,
    _create_session_matches,
)
from smart_fork.types import ChunkMatch, SessionChunk, SessionMatch


# --- Fixtures ---


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary directory for database storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "lance"


@pytest.fixture
def db(temp_db_path: Path) -> ChunkDatabase:
    """Create a database instance for testing."""
    return ChunkDatabase.open(temp_db_path)


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock embedding provider."""
    provider = MagicMock(spec=EmbeddingProvider)
    # Return a consistent 768-dim vector for any input
    provider.embed.return_value = [[0.5] * VECTOR_DIMENSIONS]
    provider.model_name.return_value = "test-model"
    return provider


@pytest.fixture
def config(temp_db_path: Path) -> SmartForkConfig:
    """Create a test configuration."""
    return SmartForkConfig(
        paths=PathsConfig(
            data_dir=temp_db_path.parent,
            lance_dir=temp_db_path.name,
        ),
        embedding=EmbeddingConfig(
            provider="ollama",
            ollama_host="http://localhost:11434",
        ),
        query=QueryConfig(
            top_chunks=20,
            top_results=5,
            recency_half_life_days=30,
        ),
    )


@pytest.fixture
def sample_chunks() -> list[SessionChunk]:
    """Create sample chunks across multiple sessions."""
    return [
        # Session 1: 2 chunks about API rate limiting
        SessionChunk(
            id="ses_api123_chunk_0",
            session_id="ses_api123",
            repo_path="/home/user/api-project",
            chunk_index=0,
            chunk_text="Implementing rate limiting for API endpoints using token bucket algorithm.",
            embedding=[0.9] + [0.1] * (VECTOR_DIMENSIONS - 1),  # High first component
            timestamp=1700000000,
            model_used="text-embedding-004",
            token_count=12,
        ),
        SessionChunk(
            id="ses_api123_chunk_1",
            session_id="ses_api123",
            repo_path="/home/user/api-project",
            chunk_index=1,
            chunk_text="Added Redis backend for distributed rate limit tracking.",
            embedding=[0.8] + [0.1] * (VECTOR_DIMENSIONS - 1),
            timestamp=1700000000,
            model_used="text-embedding-004",
            token_count=10,
        ),
        # Session 2: 1 chunk about authentication
        SessionChunk(
            id="ses_auth456_chunk_0",
            session_id="ses_auth456",
            repo_path="/home/user/auth-service",
            chunk_index=0,
            chunk_text="Implemented OAuth2 authentication flow with JWT tokens.",
            embedding=[0.3] + [0.1] * (VECTOR_DIMENSIONS - 1),  # Lower similarity
            timestamp=1700001000,
            model_used="text-embedding-004",
            token_count=9,
        ),
        # Session 3: 1 chunk about database
        SessionChunk(
            id="ses_db789_chunk_0",
            session_id="ses_db789",
            repo_path="/home/user/api-project",  # Same repo as session 1
            chunk_index=0,
            chunk_text="Optimized database queries for better performance.",
            embedding=[0.2] + [0.1] * (VECTOR_DIMENSIONS - 1),
            timestamp=1699999000,
            model_used="text-embedding-004",
            token_count=8,
        ),
    ]


@pytest.fixture
def db_with_chunks(
    db: ChunkDatabase, sample_chunks: list[SessionChunk]
) -> ChunkDatabase:
    """Create a database with sample chunks pre-loaded."""
    db.add_chunks(sample_chunks)
    return db


@pytest.fixture
def sessions_dir() -> Generator[Path, None, None]:
    """Create a temporary sessions directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# --- Query Validation Tests ---


class TestQueryValidation:
    """Tests for query input validation."""

    def test_empty_query_raises_error(
        self, config: SmartForkConfig, mock_provider: MagicMock
    ) -> None:
        """Empty query string should raise EmptyQueryError."""
        with pytest.raises(EmptyQueryError, match="cannot be empty"):
            search_sessions("", config=config, provider=mock_provider)

    def test_whitespace_only_query_raises_error(
        self, config: SmartForkConfig, mock_provider: MagicMock
    ) -> None:
        """Whitespace-only query should raise EmptyQueryError."""
        with pytest.raises(EmptyQueryError, match="cannot be empty"):
            search_sessions("   \t\n  ", config=config, provider=mock_provider)

    def test_query_is_stripped(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """Query should be stripped of leading/trailing whitespace."""
        # The mock provider will be called with stripped query
        result = search_sessions(
            "  rate limiting  ",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
        )
        # Provider should have been called with stripped text
        mock_provider.embed.assert_called_once_with(["rate limiting"])


# --- Embed Query Tests ---


class TestEmbedQuery:
    """Tests for the embed_query function."""

    def test_embed_query_returns_vector(self, mock_provider: MagicMock) -> None:
        """embed_query should return a 768-dim vector."""
        vector = embed_query("test query", provider=mock_provider)
        assert len(vector) == VECTOR_DIMENSIONS
        mock_provider.embed.assert_called_once_with(["test query"])

    def test_embed_query_empty_raises_error(self, mock_provider: MagicMock) -> None:
        """embed_query with empty string should raise EmptyQueryError."""
        with pytest.raises(EmptyQueryError):
            embed_query("", provider=mock_provider)

    def test_embed_query_strips_whitespace(self, mock_provider: MagicMock) -> None:
        """embed_query should strip whitespace from query."""
        embed_query("  test  ", provider=mock_provider)
        mock_provider.embed.assert_called_once_with(["test"])

    def test_embed_query_provider_failure_raises_query_error(
        self, mock_provider: MagicMock
    ) -> None:
        """embed_query should wrap provider errors in QueryError."""
        mock_provider.embed.side_effect = RuntimeError("API failed")
        with pytest.raises(QueryError, match="Failed to embed query"):
            embed_query("test", provider=mock_provider)


# --- Search Sessions Tests ---


class TestSearchSessions:
    """Tests for the main search_sessions function."""

    def test_search_returns_query_result(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """search_sessions should return a QueryResult object."""
        result = search_sessions(
            "rate limiting",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
        )
        assert isinstance(result, QueryResult)
        assert isinstance(result.matches, list)
        assert result.query_time_ms > 0
        assert result.total_chunks_searched >= 0

    def test_search_returns_session_matches(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """search_sessions should return SessionMatch objects."""
        result = search_sessions(
            "rate limiting",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
        )
        for match in result.matches:
            assert isinstance(match, SessionMatch)
            assert match.session_id
            assert match.repo_path
            assert match.repo_name
            assert 0.0 <= match.score <= 1.0
            assert match.chunk_count >= 1

    def test_search_respects_top_results_limit(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """search_sessions should return at most top_results matches."""
        result = search_sessions(
            "test",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
            top_results=2,
        )
        assert len(result.matches) <= 2

    def test_search_results_sorted_by_score_descending(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """search_sessions should return matches sorted by score (highest first)."""
        result = search_sessions(
            "test",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
        )
        if len(result.matches) > 1:
            scores = [m.score for m in result.matches]
            assert scores == sorted(scores, reverse=True)

    def test_search_with_repo_filter(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """search_sessions with repo_path should only return matching sessions."""
        result = search_sessions(
            "test",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
            repo_path="/home/user/api-project",
        )
        for match in result.matches:
            assert match.repo_path == "/home/user/api-project"

    def test_search_repo_filter_excludes_other_repos(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """search_sessions with repo_path should exclude other repos."""
        result = search_sessions(
            "test",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
            repo_path="/home/user/api-project",
        )
        # Should not include auth-service sessions
        session_ids = [m.session_id for m in result.matches]
        assert "ses_auth456" not in session_ids

    def test_search_empty_db_returns_empty_matches(
        self,
        config: SmartForkConfig,
        db: ChunkDatabase,  # Empty database
        mock_provider: MagicMock,
    ) -> None:
        """search_sessions on empty DB should return empty matches list."""
        result = search_sessions(
            "test",
            config=config,
            db=db,
            provider=mock_provider,
        )
        assert result.matches == []
        assert result.total_chunks_searched == 0

    def test_search_provider_failure_raises_query_error(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """search_sessions should wrap provider errors in QueryError."""
        mock_provider.embed.side_effect = RuntimeError("Embedding failed")
        with pytest.raises(QueryError, match="Failed to embed query"):
            search_sessions(
                "test",
                config=config,
                db=db_with_chunks,
                provider=mock_provider,
            )


# --- Session Grouping Tests ---


class TestSessionGrouping:
    """Tests for chunk grouping logic."""

    def test_group_chunks_by_session(self, sample_chunks: list[SessionChunk]) -> None:
        """_group_chunks_by_session should group chunks by session_id."""
        # Create ChunkMatch objects from sample chunks
        chunk_matches = [
            ChunkMatch(chunk=c, similarity=0.9 - i * 0.1, distance=0.1 + i * 0.1)
            for i, c in enumerate(sample_chunks)
        ]
        groups = _group_chunks_by_session(chunk_matches)

        # Should have 3 sessions
        assert len(groups) == 3
        assert "ses_api123" in groups
        assert "ses_auth456" in groups
        assert "ses_db789" in groups

        # ses_api123 should have 2 chunks
        assert len(groups["ses_api123"].chunks) == 2

        # ses_auth456 should have 1 chunk
        assert len(groups["ses_auth456"].chunks) == 1

    def test_session_chunk_group_best_similarity(self) -> None:
        """SessionChunkGroup.best_similarity should return max similarity."""
        group = SessionChunkGroup(
            session_id="test",
            repo_path="/test",
            timestamp=0,
            chunks=[
                ChunkMatch(
                    chunk=MagicMock(spec=SessionChunk),
                    similarity=0.7,
                    distance=0.3,
                ),
                ChunkMatch(
                    chunk=MagicMock(spec=SessionChunk),
                    similarity=0.9,
                    distance=0.1,
                ),
            ],
        )
        assert group.best_similarity == 0.9

    def test_session_chunk_group_avg_similarity(self) -> None:
        """SessionChunkGroup.avg_similarity should return mean similarity."""
        group = SessionChunkGroup(
            session_id="test",
            repo_path="/test",
            timestamp=0,
            chunks=[
                ChunkMatch(
                    chunk=MagicMock(spec=SessionChunk),
                    similarity=0.6,
                    distance=0.4,
                ),
                ChunkMatch(
                    chunk=MagicMock(spec=SessionChunk),
                    similarity=0.8,
                    distance=0.2,
                ),
            ],
        )
        assert group.avg_similarity == pytest.approx(0.7)

    def test_session_chunk_group_empty_chunks(self) -> None:
        """SessionChunkGroup with no chunks should have zero scores."""
        group = SessionChunkGroup(
            session_id="test",
            repo_path="/test",
            timestamp=0,
            chunks=[],
        )
        assert group.best_similarity == 0.0
        assert group.avg_similarity == 0.0
        assert group.best_chunk is None


# --- Result Formatting Tests ---


class TestResultFormatting:
    """Tests for session match creation with composite scoring."""

    def test_create_session_matches_returns_session_match_list(
        self,
        sample_chunks: list[SessionChunk],
        db_with_chunks: ChunkDatabase,
        config: SmartForkConfig,
        sessions_dir: Path,
    ) -> None:
        """_create_session_matches should return list of SessionMatch."""
        chunk_matches = [
            ChunkMatch(chunk=c, similarity=0.9 - i * 0.1, distance=0.1)
            for i, c in enumerate(sample_chunks)
        ]
        groups = _group_chunks_by_session(chunk_matches)
        matches = _create_session_matches(
            groups,
            top_results=5,
            db=db_with_chunks,
            config=config,
            query_timestamp=1700000000,
            sessions_dir=sessions_dir,
        )

        assert all(isinstance(m, SessionMatch) for m in matches)

    def test_create_session_matches_includes_repo_name(
        self,
        sample_chunks: list[SessionChunk],
        db_with_chunks: ChunkDatabase,
        config: SmartForkConfig,
        sessions_dir: Path,
    ) -> None:
        """SessionMatch.repo_name should be basename of repo_path."""
        chunk_matches = [
            ChunkMatch(chunk=sample_chunks[0], similarity=0.9, distance=0.1)
        ]
        groups = _group_chunks_by_session(chunk_matches)
        matches = _create_session_matches(
            groups,
            top_results=5,
            db=db_with_chunks,
            config=config,
            query_timestamp=1700000000,
            sessions_dir=sessions_dir,
        )

        assert matches[0].repo_name == "api-project"

    def test_create_session_matches_includes_snippet(
        self,
        sample_chunks: list[SessionChunk],
        db_with_chunks: ChunkDatabase,
        config: SmartForkConfig,
        sessions_dir: Path,
    ) -> None:
        """SessionMatch.best_snippet should contain chunk text."""
        chunk_matches = [
            ChunkMatch(chunk=sample_chunks[0], similarity=0.9, distance=0.1)
        ]
        groups = _group_chunks_by_session(chunk_matches)
        matches = _create_session_matches(
            groups,
            top_results=5,
            db=db_with_chunks,
            config=config,
            query_timestamp=1700000000,
            sessions_dir=sessions_dir,
        )

        assert "rate limiting" in matches[0].best_snippet.lower()

    def test_create_session_matches_truncates_long_snippets(
        self,
        db: ChunkDatabase,
        config: SmartForkConfig,
        sessions_dir: Path,
    ) -> None:
        """best_snippet should be truncated for long chunk text."""
        long_text = "x" * 300
        chunk = SessionChunk(
            id="test_chunk_0",
            session_id="test",
            repo_path="/test",
            chunk_index=0,
            chunk_text=long_text,
            embedding=[0.1] * VECTOR_DIMENSIONS,
            timestamp=1700000000,
            model_used="test",
            token_count=100,
        )
        # Add chunk to db for get_session_chunk_count
        db.add_chunks([chunk])
        chunk_matches = [ChunkMatch(chunk=chunk, similarity=0.9, distance=0.1)]
        groups = _group_chunks_by_session(chunk_matches)
        matches = _create_session_matches(
            groups,
            top_results=5,
            db=db,
            config=config,
            query_timestamp=1700000000,
            sessions_dir=sessions_dir,
        )

        # Should be truncated to around 200 chars with "..."
        assert len(matches[0].best_snippet) < 250
        assert matches[0].best_snippet.endswith("...")

    def test_create_session_matches_respects_limit(
        self,
        sample_chunks: list[SessionChunk],
        db_with_chunks: ChunkDatabase,
        config: SmartForkConfig,
        sessions_dir: Path,
    ) -> None:
        """_create_session_matches should return at most top_results."""
        chunk_matches = [
            ChunkMatch(chunk=c, similarity=0.9, distance=0.1) for c in sample_chunks
        ]
        groups = _group_chunks_by_session(chunk_matches)
        matches = _create_session_matches(
            groups,
            top_results=2,
            db=db_with_chunks,
            config=config,
            query_timestamp=1700000000,
            sessions_dir=sessions_dir,
        )

        assert len(matches) <= 2

    def test_create_session_matches_sorted_by_score(
        self,
        sample_chunks: list[SessionChunk],
        db_with_chunks: ChunkDatabase,
        config: SmartForkConfig,
        sessions_dir: Path,
    ) -> None:
        """SessionMatches should be sorted by score descending."""
        # Give each chunk a different similarity
        chunk_matches = [
            ChunkMatch(chunk=c, similarity=0.3 + i * 0.2, distance=0.1)
            for i, c in enumerate(sample_chunks)
        ]
        groups = _group_chunks_by_session(chunk_matches)
        matches = _create_session_matches(
            groups,
            top_results=10,
            db=db_with_chunks,
            config=config,
            query_timestamp=1700000000,
            sessions_dir=sessions_dir,
        )

        scores = [m.score for m in matches]
        assert scores == sorted(scores, reverse=True)


# --- Performance Tests ---


class TestQueryPerformance:
    """Tests for query performance characteristics."""

    def test_query_completes_within_timeout(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """Query should complete within 3 seconds (spec requirement)."""
        start = time.perf_counter()
        search_sessions(
            "test query",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0  # 3 second max latency per spec

    def test_query_time_reported_in_result(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """QueryResult should include query_time_ms."""
        result = search_sessions(
            "test",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
        )
        assert result.query_time_ms > 0


# --- Integration Tests ---


class TestQueryIntegration:
    """Integration tests for the full query flow."""

    def test_end_to_end_search_with_mock_provider(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """Full search flow should work with mock provider."""
        # Configure mock to return a vector similar to api123 chunks
        mock_provider.embed.return_value = [[0.9] + [0.1] * (VECTOR_DIMENSIONS - 1)]

        result = search_sessions(
            "rate limiting API",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
        )

        # Should find matches
        assert len(result.matches) > 0
        # Best match should be the API session (most similar embedding)
        assert result.matches[0].session_id == "ses_api123"

    def test_chunk_count_aggregates_correctly(
        self,
        config: SmartForkConfig,
        db_with_chunks: ChunkDatabase,
        mock_provider: MagicMock,
    ) -> None:
        """chunk_count should reflect number of matching chunks per session."""
        # Configure mock to return a vector that matches multiple chunks
        mock_provider.embed.return_value = [[0.85] + [0.1] * (VECTOR_DIMENSIONS - 1)]

        result = search_sessions(
            "rate limiting",
            config=config,
            db=db_with_chunks,
            provider=mock_provider,
            top_k=10,  # Get enough chunks to see aggregation
        )

        # Find the api123 session
        api_match = next(
            (m for m in result.matches if m.session_id == "ses_api123"), None
        )
        assert api_match is not None
        # Should have 2 chunks aggregated
        assert api_match.chunk_count == 2

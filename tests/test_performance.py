"""Performance tests for Smart Fork.

Validates that the system meets performance requirements:
- Ingestion: 100 sessions < 10 minutes
- Search: Query latency < 3 seconds

These tests use mock embeddings to isolate pipeline performance
from external API latency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from smart_fork.chunker import Chunker
from smart_fork.config import (
    ChunkingConfig,
    EmbeddingConfig,
    PathsConfig,
    SmartForkConfig,
)
from smart_fork.db import ChunkDatabase
from smart_fork.ingest import sync_sessions
from smart_fork.query import search_sessions
from smart_fork.types import SessionChunk


# ============================================================================
# Test Constants
# ============================================================================

# Performance thresholds
INGESTION_100_SESSIONS_MAX_SECONDS = 600  # 10 minutes
SEARCH_LATENCY_MAX_SECONDS = 3.0

# Realistic session sizes (based on actual OpenCode usage patterns)
MESSAGES_PER_SESSION_MIN = 4
MESSAGES_PER_SESSION_MAX = 40


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def perf_sessions_dir(tmp_path: Path) -> Path:
    """Create a temporary sessions directory for performance testing."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions


@pytest.fixture
def perf_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for LanceDB and config."""
    data = tmp_path / "smart-fork"
    data.mkdir()
    return data


def generate_realistic_message(role: str, index: int, topic: str) -> str:
    """Generate a realistic message with varied content.

    Creates messages that resemble actual coding assistance conversations
    to test chunking and embedding performance realistically.
    """
    if role == "user":
        templates = [
            f"I'm working on {topic} and need help with implementing a feature. "
            f"The current code has issues with error handling and I want to refactor it. "
            f"Can you help me understand the best approach for this task? "
            f"Here's what I've tried so far: I started by looking at the existing implementation "
            f"but found it difficult to extend. The main problem seems to be tight coupling.",
            f"How do I add {topic} to my project? I've been reading the documentation "
            f"but I'm confused about the configuration options. Should I use the default settings "
            f"or customize them for my use case? I'm particularly interested in performance.",
            f"Can you help me debug this {topic} issue? The error message says something about "
            f"type mismatch but I can't figure out where the problem is. I've tried several "
            f"approaches including checking the input types and adding validation.",
            f"I want to implement {topic} following best practices. What patterns should I use? "
            f"I've seen examples using both factory pattern and dependency injection. "
            f"Which one would be more appropriate for a project of this size?",
            f"My {topic} tests are failing. The test output shows assertion errors "
            f"but the logic looks correct to me. Could you review this and suggest fixes? "
            f"I think the problem might be in the setup or teardown methods.",
        ]
    else:
        templates = [
            f"I'd be happy to help you with {topic}. Let me break down the solution into steps. "
            f"First, you'll want to ensure proper error handling by wrapping the main logic "
            f"in try-except blocks. Then, consider extracting the core functionality into "
            f"separate functions for better testability. Here's an example implementation: "
            f"```python\ndef process_data(input_data):\n    try:\n        result = transform(input_data)\n        return result\n    except ValueError as e:\n        logger.error(f'Processing failed: {{e}}')\n        raise\n```",
            f"For {topic}, I recommend the following approach. The key insight is that "
            f"you need to separate concerns properly. The configuration should be loaded "
            f"at startup and passed to components that need it. This makes testing easier "
            f"and reduces coupling between modules. Let me show you the pattern: "
            f"Create a config dataclass, load it from environment or file, inject into services.",
            f"Looking at your {topic} issue, I can see the problem. The type mismatch occurs "
            f"because the function expects a list but receives a string. You can fix this by "
            f"adding type validation at the entry point. Here's a decorator that helps: "
            f"```python\ndef validate_types(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        # Type checking logic\n        return func(*args, **kwargs)\n    return wrapper\n```",
            f"For implementing {topic} with best practices, I'd suggest using the repository "
            f"pattern combined with dependency injection. This gives you flexibility and "
            f"testability. The factory pattern is useful for creating complex objects, but "
            f"for services, DI is cleaner. Here's how to structure it: define interfaces, "
            f"implement concrete classes, wire dependencies at composition root.",
            f"I've analyzed your {topic} test failures. The issue is that the test setup "
            f"doesn't initialize the database connection properly. The fixture should use "
            f"a context manager to ensure cleanup. Try updating your conftest.py with: "
            f"```python\n@pytest.fixture\ndef db_connection():\n    conn = create_connection()\n    yield conn\n    conn.close()\n```",
        ]

    return templates[index % len(templates)]


def create_synthetic_session(
    sessions_dir: Path,
    session_id: str,
    num_messages: int,
    topic: str,
    timestamp: int,
    parent_session_id: str | None = None,
) -> Path:
    """Create a synthetic session with realistic conversation content."""
    session_path = sessions_dir / session_id
    session_path.mkdir()

    # Create session.json
    session_data: dict[str, Any] = {
        "working_directory": f"/home/user/projects/{topic.replace(' ', '-')}",
        "timestamp": timestamp,
        "model": "claude-3-opus",
    }
    if parent_session_id:
        session_data["parent_session_id"] = parent_session_id

    (session_path / "session.json").write_text(json.dumps(session_data))

    # Create messages.json with alternating user/assistant messages
    messages = []
    for i in range(num_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = generate_realistic_message(role, i, topic)
        messages.append({"role": role, "content": content})

    (session_path / "messages.json").write_text(json.dumps(messages))

    return session_path


def create_mock_embeddings(dimensions: int = 768) -> list[float]:
    """Create mock embeddings for testing."""
    import random

    return [random.random() for _ in range(dimensions)]


def create_test_config(sessions_dir: Path, data_dir: Path) -> SmartForkConfig:
    """Create a test configuration pointing to temp directories."""
    return SmartForkConfig(
        paths=PathsConfig(
            sessions_dir=sessions_dir,
            data_dir=data_dir,
        ),
        embedding=EmbeddingConfig(
            provider="ollama",  # Will be mocked
            rate_limit_ms=0,  # No rate limiting for performance test
        ),
    )


# ============================================================================
# Performance Tests
# ============================================================================


class TestIngestionPerformance:
    """Tests for ingestion pipeline performance."""

    def test_ingestion_100_sessions_under_10_minutes(
        self, perf_sessions_dir: Path, perf_data_dir: Path
    ) -> None:
        """Validate that ingesting 100 sessions completes in under 10 minutes.

        This test uses mock embeddings to isolate ingestion pipeline performance
        from external API latency. With real embeddings, the time would be dominated
        by API calls (rate limited at 100ms between batches).

        Success criteria:
        - 100 sessions ingested without errors
        - Total time < 600 seconds (10 minutes)
        - All chunks stored in LanceDB
        """
        # Create 100 synthetic sessions with varied content
        topics = [
            "API development",
            "database migrations",
            "authentication flow",
            "caching strategy",
            "error handling",
            "logging system",
            "testing framework",
            "deployment pipeline",
            "monitoring setup",
            "configuration management",
            "dependency injection",
            "event sourcing",
            "microservices",
            "GraphQL schema",
            "REST endpoints",
            "WebSocket handling",
            "batch processing",
            "async patterns",
            "type safety",
            "code refactoring",
        ]

        base_timestamp = 1700000000
        parent_id = None

        for i in range(100):
            session_id = f"ses_perf{i:03d}"
            topic = topics[i % len(topics)]
            # Vary message count between sessions (4-40 messages)
            num_messages = MESSAGES_PER_SESSION_MIN + (
                i % (MESSAGES_PER_SESSION_MAX - MESSAGES_PER_SESSION_MIN + 1)
            )
            timestamp = base_timestamp + (i * 3600)  # 1 hour apart

            # Create some forked sessions (every 5th session forks from previous)
            if i > 0 and i % 5 == 0:
                parent_id = f"ses_perf{i - 1:03d}"
            else:
                parent_id = None

            create_synthetic_session(
                perf_sessions_dir,
                session_id,
                num_messages,
                topic,
                timestamp,
                parent_id,
            )

        # Create config pointing to test directories
        config = create_test_config(perf_sessions_dir, perf_data_dir)

        # Mock embedding provider to return consistent mock embeddings
        mock_provider = MagicMock()
        mock_provider.embed.side_effect = lambda texts: [
            create_mock_embeddings() for _ in texts
        ]
        mock_provider.model_name.return_value = "mock-embedding-model"

        # Track ingestion time
        start_time = time.perf_counter()

        with patch("smart_fork.ingest.create_provider", return_value=mock_provider):
            result = sync_sessions(config, force=True)

        end_time = time.perf_counter()
        elapsed_seconds = end_time - start_time

        # Verify results
        assert result.sessions_processed == 100, (
            f"Expected 100 sessions, got {result.sessions_processed}"
        )
        assert result.sessions_failed == 0, (
            f"Expected 0 failures, got {result.sessions_failed}"
        )
        assert result.chunks_added > 0, (
            f"Expected chunks to be added, got {result.chunks_added}"
        )

        # Performance assertion
        assert elapsed_seconds < INGESTION_100_SESSIONS_MAX_SECONDS, (
            f"Ingestion took {elapsed_seconds:.2f}s, exceeds limit of "
            f"{INGESTION_100_SESSIONS_MAX_SECONDS}s (10 minutes)"
        )

        # Log performance metrics for visibility
        print(f"\n=== Ingestion Performance Results ===")
        print(f"Sessions: {result.sessions_processed}")
        print(f"Chunks: {result.chunks_added}")
        print(f"Time: {elapsed_seconds:.2f}s")
        print(f"Rate: {result.sessions_processed / elapsed_seconds:.2f} sessions/sec")
        print(f"Threshold: {INGESTION_100_SESSIONS_MAX_SECONDS}s (10 min)")


class TestSearchPerformance:
    """Tests for search query performance."""

    def test_search_latency_under_3_seconds(
        self, perf_sessions_dir: Path, perf_data_dir: Path
    ) -> None:
        """Validate that search queries complete in under 3 seconds.

        This test pre-populates LanceDB with chunks and measures query latency.
        Uses mock embeddings for the query to isolate search performance.

        Success criteria:
        - Search returns results
        - Query latency < 3 seconds
        - Results are properly scored and ranked
        """
        # Set up database with sample chunks
        lance_dir = perf_data_dir / "lance"
        lance_dir.mkdir(parents=True, exist_ok=True)
        db = ChunkDatabase.open(lance_dir)

        # Create 500 chunks (simulating ~50-100 sessions)
        chunks: list[SessionChunk] = []
        topics = [
            "webhook handling",
            "authentication",
            "rate limiting",
            "database queries",
            "API endpoints",
            "error handling",
            "logging",
            "testing",
            "deployment",
            "monitoring",
        ]

        for i in range(500):
            session_idx = i // 5  # 5 chunks per session
            topic = topics[session_idx % len(topics)]

            chunk = SessionChunk(
                id=f"ses_search{session_idx:03d}_chunk_{i % 5}",
                session_id=f"ses_search{session_idx:03d}",
                repo_path=f"/home/user/projects/{topic.replace(' ', '-')}",
                chunk_index=i % 5,
                chunk_text=f"Working on {topic} implementation with detailed discussion of patterns and best practices for production systems.",
                embedding=create_mock_embeddings(),
                timestamp=1700000000 + (session_idx * 3600),
                model_used="mock-embedding-model",
                token_count=50,
            )
            chunks.append(chunk)

        db.add_chunks(chunks)

        # Create config
        config = create_test_config(perf_sessions_dir, perf_data_dir)

        # Mock embedding provider for query
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [create_mock_embeddings()]
        mock_provider.model_name.return_value = "mock-embedding-model"

        # Measure search latency over multiple queries
        queries = [
            "webhook handling patterns",
            "authentication flow implementation",
            "database query optimization",
            "error handling best practices",
            "API endpoint design",
        ]

        latencies: list[float] = []

        with patch("smart_fork.query.create_provider", return_value=mock_provider):
            for query in queries:
                start_time = time.perf_counter()
                result = search_sessions(query, config=config)
                end_time = time.perf_counter()

                latencies.append(end_time - start_time)

                # Verify results returned
                assert len(result.matches) > 0, f"Query '{query}' returned no results"

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        # Performance assertions
        assert max_latency < SEARCH_LATENCY_MAX_SECONDS, (
            f"Max search latency {max_latency:.3f}s exceeds limit of "
            f"{SEARCH_LATENCY_MAX_SECONDS}s"
        )

        assert avg_latency < SEARCH_LATENCY_MAX_SECONDS, (
            f"Avg search latency {avg_latency:.3f}s exceeds limit of "
            f"{SEARCH_LATENCY_MAX_SECONDS}s"
        )

        # Log performance metrics
        print(f"\n=== Search Performance Results ===")
        print(f"Queries: {len(queries)}")
        print(f"Database chunks: 500")
        print(f"Avg latency: {avg_latency:.3f}s")
        print(f"Max latency: {max_latency:.3f}s")
        print(f"Min latency: {min(latencies):.3f}s")
        print(f"Threshold: {SEARCH_LATENCY_MAX_SECONDS}s")


class TestScalabilityMetrics:
    """Tests to validate scalability assumptions."""

    def test_chunking_throughput(self) -> None:
        """Measure chunking throughput for large sessions.

        Creates a large session (1000+ messages) and measures chunking speed.
        This validates that the chunking pipeline isn't a bottleneck.
        """
        # Generate large text (simulating a very long session)
        large_messages = []
        for i in range(200):  # 200 messages
            role = "user" if i % 2 == 0 else "assistant"
            content = generate_realistic_message(role, i, "comprehensive refactoring")
            large_messages.append(f"{role}: {content}")

        large_text = "\n\n".join(large_messages)

        # Create chunker
        chunker = Chunker(ChunkingConfig(target_tokens=512, overlap_tokens=50))

        # Measure chunking time
        start_time = time.perf_counter()
        chunks = chunker.chunk_text(large_text)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        # Should chunk quickly (< 1 second for any reasonable size)
        assert elapsed_ms < 1000, f"Chunking took {elapsed_ms:.2f}ms, expected < 1000ms"

        print(f"\n=== Chunking Performance Results ===")
        print(f"Input text length: {len(large_text):,} chars")
        print(f"Chunks produced: {len(chunks)}")
        print(f"Time: {elapsed_ms:.2f}ms")
        print(f"Throughput: {len(large_text) / (elapsed_ms / 1000):,.0f} chars/sec")

    def test_database_write_throughput(self, perf_data_dir: Path) -> None:
        """Measure LanceDB write throughput.

        Tests bulk insertion of chunks to validate database isn't a bottleneck.
        """
        lance_dir = perf_data_dir / "lance"
        lance_dir.mkdir(parents=True, exist_ok=True)
        db = ChunkDatabase.open(lance_dir)

        # Create 1000 chunks
        chunks: list[SessionChunk] = []
        for i in range(1000):
            chunk = SessionChunk(
                id=f"ses_write{i:04d}_chunk_0",
                session_id=f"ses_write{i:04d}",
                repo_path="/home/user/test-project",
                chunk_index=0,
                chunk_text=f"Test chunk content {i} with some additional text for realism.",
                embedding=create_mock_embeddings(),
                timestamp=1700000000 + i,
                model_used="mock-model",
                token_count=20,
            )
            chunks.append(chunk)

        # Measure write time
        start_time = time.perf_counter()
        db.add_chunks(chunks)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        # Should write quickly (< 5 seconds for 1000 chunks)
        assert elapsed_ms < 5000, f"Writing took {elapsed_ms:.2f}ms, expected < 5000ms"

        # Verify writes succeeded
        count = db.count_chunks()
        assert count == 1000, f"Expected 1000 chunks, got {count}"

        print(f"\n=== Database Write Performance Results ===")
        print(f"Chunks written: 1000")
        print(f"Time: {elapsed_ms:.2f}ms")
        print(f"Throughput: {1000 / (elapsed_ms / 1000):.0f} chunks/sec")

"""Tests for the LanceDB wrapper module.

Tests cover:
- Database creation and opening
- Table creation with correct schema
- Adding and retrieving chunks
- Counting operations
- Deleting by session
- Error handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from smart_fork.db import (
    SESSIONS_TABLE,
    VECTOR_DIMENSIONS,
    ChunkDatabase,
    DatabaseError,
    SchemaError,
    TableNotFoundError,
    _get_arrow_schema,
)
from smart_fork.types import SessionChunk


# --- Fixtures ---


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary directory for database storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "lance"


@pytest.fixture
def db(temp_db_path: Path) -> ChunkDatabase:
    """Create a database instance for testing."""
    return ChunkDatabase.open(temp_db_path)


@pytest.fixture
def sample_chunk() -> SessionChunk:
    """Create a sample session chunk for testing."""
    return SessionChunk(
        id="ses_test123_chunk_0",
        session_id="ses_test123",
        repo_path="/home/user/project",
        chunk_index=0,
        chunk_text="This is a test chunk with some content about testing.",
        embedding=[0.1] * VECTOR_DIMENSIONS,  # 768-dim vector
        timestamp=1700000000,
        model_used="text-embedding-004",
        token_count=12,
    )


@pytest.fixture
def sample_chunks() -> list[SessionChunk]:
    """Create multiple sample chunks across sessions."""
    return [
        SessionChunk(
            id="ses_abc_chunk_0",
            session_id="ses_abc",
            repo_path="/home/user/project-a",
            chunk_index=0,
            chunk_text="First chunk of session abc.",
            embedding=[0.1 + i * 0.01 for i in range(VECTOR_DIMENSIONS)],
            timestamp=1700000000,
            model_used="text-embedding-004",
            token_count=6,
        ),
        SessionChunk(
            id="ses_abc_chunk_1",
            session_id="ses_abc",
            repo_path="/home/user/project-a",
            chunk_index=1,
            chunk_text="Second chunk of session abc.",
            embedding=[0.2 + i * 0.01 for i in range(VECTOR_DIMENSIONS)],
            timestamp=1700000000,
            model_used="text-embedding-004",
            token_count=6,
        ),
        SessionChunk(
            id="ses_def_chunk_0",
            session_id="ses_def",
            repo_path="/home/user/project-b",
            chunk_index=0,
            chunk_text="First chunk of session def.",
            embedding=[0.3 + i * 0.01 for i in range(VECTOR_DIMENSIONS)],
            timestamp=1700001000,
            model_used="nomic-embed-text",
            token_count=6,
        ),
    ]


# --- Schema Tests ---


class TestArrowSchema:
    """Tests for the Arrow schema definition."""

    def test_schema_has_all_fields(self) -> None:
        """Schema includes all SessionChunk fields."""
        schema = _get_arrow_schema()
        field_names = [f.name for f in schema]
        expected = [
            "id",
            "session_id",
            "repo_path",
            "chunk_index",
            "chunk_text",
            "embedding",
            "timestamp",
            "model_used",
            "token_count",
        ]
        assert field_names == expected

    def test_embedding_field_is_fixed_size_list(self) -> None:
        """Embedding field is a fixed-size list of float32."""
        import pyarrow as pa

        schema = _get_arrow_schema()
        embedding_field = schema.field("embedding")
        assert pa.types.is_fixed_size_list(embedding_field.type)
        assert embedding_field.type.list_size == VECTOR_DIMENSIONS
        assert pa.types.is_float32(embedding_field.type.value_type)


# --- Database Creation Tests ---


class TestDatabaseOpen:
    """Tests for database opening and creation."""

    def test_open_creates_directory(self, temp_db_path: Path) -> None:
        """Opening a database creates the directory if needed."""
        assert not temp_db_path.exists()
        ChunkDatabase.open(temp_db_path)
        assert temp_db_path.exists()

    def test_open_with_home_expansion(self) -> None:
        """Paths with ~ are expanded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a path that would need expansion
            db_path = Path(tmpdir) / "test_lance"
            db = ChunkDatabase.open(db_path)
            assert db.path == db_path

    def test_path_property_returns_db_location(self, db: ChunkDatabase) -> None:
        """The path property returns the database directory."""
        assert db.path.exists()

    def test_reopen_existing_database(self, temp_db_path: Path) -> None:
        """Can reopen an existing database."""
        # Create database with a table
        db1 = ChunkDatabase.open(temp_db_path)
        db1.create_table()
        db1.close()

        # Reopen and verify table exists
        db2 = ChunkDatabase.open(temp_db_path)
        assert db2.table_exists()


# --- Table Operations Tests ---


class TestTableOperations:
    """Tests for table creation and management."""

    def test_table_exists_false_initially(self, db: ChunkDatabase) -> None:
        """Table doesn't exist in a fresh database."""
        assert not db.table_exists()

    def test_create_table_creates_sessions_table(self, db: ChunkDatabase) -> None:
        """create_table creates the sessions table."""
        db.create_table()
        assert db.table_exists()

    def test_create_table_is_idempotent(self, db: ChunkDatabase) -> None:
        """Calling create_table twice doesn't raise an error."""
        db.create_table()
        db.create_table()  # Should not raise
        assert db.table_exists()

    def test_get_table_raises_when_not_exists(self, db: ChunkDatabase) -> None:
        """get_table raises TableNotFoundError when table doesn't exist."""
        with pytest.raises(TableNotFoundError) as exc_info:
            db.get_table()
        assert SESSIONS_TABLE in str(exc_info.value)
        assert "sync" in str(exc_info.value).lower()

    def test_get_table_returns_table(self, db: ChunkDatabase) -> None:
        """get_table returns the table object when it exists."""
        db.create_table()
        table = db.get_table()
        assert table is not None

    def test_drop_table_removes_table(
        self, db: ChunkDatabase, sample_chunk: SessionChunk
    ) -> None:
        """drop_table removes the sessions table."""
        db.add_chunks([sample_chunk])
        assert db.table_exists()

        result = db.drop_table()

        assert result is True
        assert not db.table_exists()

    def test_drop_table_returns_false_when_not_exists(self, db: ChunkDatabase) -> None:
        """drop_table returns False if table doesn't exist."""
        result = db.drop_table()
        assert result is False


# --- Chunk CRUD Tests ---


class TestAddChunks:
    """Tests for adding chunks to the database."""

    def test_add_single_chunk(
        self, db: ChunkDatabase, sample_chunk: SessionChunk
    ) -> None:
        """Can add a single chunk to the database."""
        count = db.add_chunks([sample_chunk])
        assert count == 1
        assert db.count_chunks() == 1

    def test_add_multiple_chunks(
        self, db: ChunkDatabase, sample_chunks: list[SessionChunk]
    ) -> None:
        """Can add multiple chunks at once."""
        count = db.add_chunks(sample_chunks)
        assert count == 3
        assert db.count_chunks() == 3

    def test_add_empty_list_returns_zero(self, db: ChunkDatabase) -> None:
        """Adding empty list returns 0 and doesn't create table."""
        count = db.add_chunks([])
        assert count == 0
        assert not db.table_exists()

    def test_add_chunks_creates_table(
        self, db: ChunkDatabase, sample_chunk: SessionChunk
    ) -> None:
        """Adding chunks creates table if it doesn't exist."""
        assert not db.table_exists()
        db.add_chunks([sample_chunk])
        assert db.table_exists()

    def test_add_chunks_appends_by_default(
        self, db: ChunkDatabase, sample_chunks: list[SessionChunk]
    ) -> None:
        """Adding chunks appends to existing data by default."""
        db.add_chunks(sample_chunks[:1])
        assert db.count_chunks() == 1

        db.add_chunks(sample_chunks[1:2])
        assert db.count_chunks() == 2

    def test_add_chunks_overwrite_mode(
        self, db: ChunkDatabase, sample_chunks: list[SessionChunk]
    ) -> None:
        """Adding chunks with mode='overwrite' replaces all data."""
        db.add_chunks(sample_chunks[:2])
        assert db.count_chunks() == 2

        db.add_chunks(sample_chunks[2:], mode="overwrite")
        assert db.count_chunks() == 1

    def test_add_chunk_wrong_embedding_size_raises(self, db: ChunkDatabase) -> None:
        """Adding chunk with wrong embedding dimensions raises SchemaError."""
        bad_chunk = SessionChunk(
            id="bad_chunk",
            session_id="ses_bad",
            repo_path="/tmp",
            chunk_index=0,
            chunk_text="Bad embedding size",
            embedding=[0.1] * 100,  # Wrong size!
            timestamp=1700000000,
            model_used="test",
            token_count=3,
        )
        with pytest.raises(SchemaError) as exc_info:
            db.add_chunks([bad_chunk])
        assert "100" in str(exc_info.value)
        assert str(VECTOR_DIMENSIONS) in str(exc_info.value)


# --- Count Operations Tests ---


class TestCounting:
    """Tests for counting operations."""

    def test_count_chunks_zero_when_empty(self, db: ChunkDatabase) -> None:
        """count_chunks returns 0 when database is empty."""
        assert db.count_chunks() == 0

    def test_count_chunks_returns_total(
        self, db: ChunkDatabase, sample_chunks: list[SessionChunk]
    ) -> None:
        """count_chunks returns total number of chunks."""
        db.add_chunks(sample_chunks)
        assert db.count_chunks() == 3

    def test_count_sessions_zero_when_empty(self, db: ChunkDatabase) -> None:
        """count_sessions returns 0 when database is empty."""
        assert db.count_sessions() == 0

    def test_count_sessions_returns_unique_count(
        self, db: ChunkDatabase, sample_chunks: list[SessionChunk]
    ) -> None:
        """count_sessions returns number of unique session IDs."""
        db.add_chunks(sample_chunks)
        # sample_chunks has 2 chunks from ses_abc and 1 from ses_def
        assert db.count_sessions() == 2


# --- Delete Operations Tests ---


class TestDeleteBySession:
    """Tests for deleting chunks by session ID."""

    def test_delete_removes_all_session_chunks(
        self, db: ChunkDatabase, sample_chunks: list[SessionChunk]
    ) -> None:
        """delete_by_session removes all chunks for a session."""
        db.add_chunks(sample_chunks)
        assert db.count_chunks() == 3

        deleted = db.delete_by_session("ses_abc")

        assert deleted == 2
        assert db.count_chunks() == 1

    def test_delete_nonexistent_session_returns_zero(
        self, db: ChunkDatabase, sample_chunks: list[SessionChunk]
    ) -> None:
        """delete_by_session returns 0 for nonexistent session."""
        db.add_chunks(sample_chunks)
        deleted = db.delete_by_session("ses_nonexistent")
        assert deleted == 0
        assert db.count_chunks() == 3

    def test_delete_when_no_table_returns_zero(self, db: ChunkDatabase) -> None:
        """delete_by_session returns 0 when table doesn't exist."""
        deleted = db.delete_by_session("ses_any")
        assert deleted == 0


# --- Connection Management Tests ---


class TestConnectionManagement:
    """Tests for database connection handling."""

    def test_close_allows_garbage_collection(
        self, temp_db_path: Path, sample_chunk: SessionChunk
    ) -> None:
        """Closing the database allows the connection to be garbage collected."""
        db = ChunkDatabase.open(temp_db_path)
        db.add_chunks([sample_chunk])
        db.close()
        # After close, connection should be None
        assert db._connection is None  # noqa: SLF001

    def test_operations_after_close_raise(
        self, temp_db_path: Path, sample_chunk: SessionChunk
    ) -> None:
        """Operations after close raise appropriate errors."""
        db = ChunkDatabase.open(temp_db_path)
        db.add_chunks([sample_chunk])
        db.close()

        # Attempting operations should fail
        with pytest.raises((DatabaseError, AttributeError, TypeError)):
            db.count_chunks()


# --- Schema Validation Tests ---


class TestSchemaValidation:
    """Tests for schema validation on table creation."""

    def test_create_table_validates_existing_schema(self, temp_db_path: Path) -> None:
        """create_table validates schema matches when table exists."""
        import lancedb
        import pyarrow as pa

        # Create a table with wrong schema directly
        conn = lancedb.connect(str(temp_db_path))
        wrong_schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("wrong_field", pa.string()),  # Different field
            ]
        )
        conn.create_table(SESSIONS_TABLE, schema=wrong_schema)

        # Opening with our wrapper should detect mismatch
        db = ChunkDatabase.open(temp_db_path)
        with pytest.raises(SchemaError) as exc_info:
            db.create_table()
        assert (
            "wrong_field" in str(exc_info.value)
            or "schema" in str(exc_info.value).lower()
        )


# --- Error Handling Tests ---


class TestErrorHandling:
    """Tests for error handling in edge cases."""

    def test_database_error_on_invalid_path(self) -> None:
        """Opening database at invalid path raises DatabaseError."""
        # This should fail because /proc is not writable
        invalid_path = Path("/proc/sys/kernel/nonexistent/lance")
        with pytest.raises(DatabaseError):
            ChunkDatabase.open(invalid_path)

    def test_table_not_found_error_message(self, db: ChunkDatabase) -> None:
        """TableNotFoundError has helpful message."""
        with pytest.raises(TableNotFoundError) as exc_info:
            db.get_table()
        error_msg = str(exc_info.value)
        assert "sessions" in error_msg.lower()
        assert "sync" in error_msg.lower()

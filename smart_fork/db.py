"""LanceDB wrapper for Smart Fork vector storage.

Provides a thin wrapper around LanceDB for storing and querying session chunks.
The database is stored at ~/.local/share/opencode/smart-fork/lance/ by default.

Key design decisions:
- Uses SessionChunk dataclass directly as schema (LanceDB infers Arrow schema)
- Table is created lazily on first write operation
- Vector column uses 768 dimensions (matches embedding providers)
- All operations are synchronous (LanceDB is embedded, no network latency)

Schema versioning note:
LanceDB doesn't support schema migrations. If schema changes, the database
must be recreated. This is acceptable for v1 since data can be regenerated
from source sessions via `smart-fork sync --force`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import lancedb  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]

from smart_fork.types import ChunkMatch, SessionChunk

if TYPE_CHECKING:
    from lancedb.table import Table as LanceTable  # type: ignore[import-untyped]


# Table name for session chunks
SESSIONS_TABLE = "sessions"

# Expected embedding dimensions (must match embedding providers)
VECTOR_DIMENSIONS = 768


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


class TableNotFoundError(DatabaseError):
    """Raised when attempting to query a table that doesn't exist."""

    pass


class SchemaError(DatabaseError):
    """Raised when there's a schema mismatch or validation error."""

    pass


def _get_arrow_schema() -> pa.Schema:
    """Build the Arrow schema for the sessions table.

    Returns:
        PyArrow schema matching SessionChunk dataclass.

    Why explicit schema:
    - Ensures vector column has correct fixed-size list type for ANN search
    - Provides consistent schema even before first data is written
    - Enables schema validation on insert
    """
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("session_id", pa.string()),
            pa.field("repo_path", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("chunk_text", pa.string()),
            pa.field(
                "embedding",
                pa.list_(pa.float32(), VECTOR_DIMENSIONS),
            ),
            pa.field("timestamp", pa.int64()),
            pa.field("model_used", pa.string()),
            pa.field("token_count", pa.int32()),
        ]
    )


class ChunkDatabase:
    """LanceDB wrapper for session chunk storage and retrieval.

    Provides methods for:
    - Opening/creating the database at the configured path
    - Adding chunks to the sessions table
    - Vector search with optional repo filtering
    - Deleting chunks by session_id

    The database is embedded (no server required) and stored locally.

    Example:
        db = ChunkDatabase.open(Path("~/.local/share/opencode/smart-fork/lance"))
        db.add_chunks([chunk1, chunk2, chunk3])
        results = db.search(query_vector, top_k=10)
    """

    def __init__(self, connection: lancedb.DBConnection, path: Path) -> None:
        """Initialize database wrapper.

        Args:
            connection: Open LanceDB connection.
            path: Path to the database directory.

        Note:
            Use ChunkDatabase.open() factory method instead of direct instantiation.
        """
        self._connection = connection
        self._path = path

    @classmethod
    def open(cls, path: Path) -> ChunkDatabase:
        """Open or create a LanceDB database at the given path.

        Creates the directory and database if they don't exist.

        Args:
            path: Directory path for LanceDB storage.

        Returns:
            ChunkDatabase instance ready for operations.

        Raises:
            DatabaseError: If database cannot be opened or created.
        """
        try:
            # Ensure directory exists
            path = path.expanduser()
            path.mkdir(parents=True, exist_ok=True)

            # Open database connection
            connection = lancedb.connect(str(path))
            return cls(connection, path)
        except Exception as e:
            raise DatabaseError(f"Failed to open database at {path}: {e}") from e

    @property
    def path(self) -> Path:
        """Return the database directory path."""
        return self._path

    def table_exists(self) -> bool:
        """Check if the sessions table exists.

        Returns:
            True if the sessions table exists, False otherwise.
        """
        response = self._connection.list_tables()
        # LanceDB returns ListTablesResponse with .tables attribute
        table_names: list[str] = response.tables
        return SESSIONS_TABLE in table_names

    def get_table(self) -> LanceTable:
        """Get the sessions table.

        Returns:
            LanceDB table object.

        Raises:
            TableNotFoundError: If sessions table doesn't exist.
        """
        if not self.table_exists():
            raise TableNotFoundError(
                f"Table '{SESSIONS_TABLE}' does not exist. "
                "Run 'smart-fork sync' to index sessions first."
            )
        return self._connection.open_table(SESSIONS_TABLE)

    def create_table(self) -> LanceTable:
        """Create the sessions table with the correct schema.

        Creates an empty table if it doesn't exist. If table exists,
        returns the existing table (no-op).

        Returns:
            LanceDB table object.

        Raises:
            SchemaError: If existing table has incompatible schema.
        """
        if self.table_exists():
            # Validate existing schema matches expected
            table = self._connection.open_table(SESSIONS_TABLE)
            existing_schema = table.schema
            expected_schema = _get_arrow_schema()

            # Compare field names and types (not metadata)
            existing_fields = {f.name: f.type for f in existing_schema}
            expected_fields = {f.name: f.type for f in expected_schema}

            if existing_fields != expected_fields:
                raise SchemaError(
                    f"Existing table schema doesn't match expected schema. "
                    f"Expected fields: {list(expected_fields.keys())}, "
                    f"Got fields: {list(existing_fields.keys())}. "
                    f"Run 'smart-fork sync --force' to recreate the database."
                )
            return table

        # Create empty table with schema
        # LanceDB requires at least one record to create a table, so we use
        # create_table with explicit schema
        return self._connection.create_table(
            SESSIONS_TABLE,
            schema=_get_arrow_schema(),
        )

    def add_chunks(
        self,
        chunks: list[SessionChunk],
        *,
        mode: Literal["append", "overwrite"] = "append",
    ) -> int:
        """Add session chunks to the database.

        Args:
            chunks: List of SessionChunk objects to insert.
            mode: Write mode - "append" (default) or "overwrite".
                  Use "overwrite" with caution as it replaces all data.

        Returns:
            Number of chunks added.

        Raises:
            SchemaError: If chunk data doesn't match schema.
            DatabaseError: If write operation fails.

        Example:
            chunks = [SessionChunk(id="ses_123_chunk_0", ...)]
            count = db.add_chunks(chunks)
        """
        if not chunks:
            return 0

        # Convert dataclasses to dicts
        records = [asdict(chunk) for chunk in chunks]

        # Validate embedding dimensions
        for i, record in enumerate(records):
            embedding = record.get("embedding", [])
            if len(embedding) != VECTOR_DIMENSIONS:
                raise SchemaError(
                    f"Chunk {i} has embedding with {len(embedding)} dimensions, "
                    f"expected {VECTOR_DIMENSIONS}"
                )

        try:
            if self.table_exists():
                table = self._connection.open_table(SESSIONS_TABLE)
                table.add(records, mode=mode)
            else:
                # Create table with first batch of data
                self._connection.create_table(
                    SESSIONS_TABLE,
                    data=records,
                    schema=_get_arrow_schema(),
                )
            return len(chunks)
        except Exception as e:
            raise DatabaseError(f"Failed to add chunks: {e}") from e

    def delete_by_session(self, session_id: str) -> int:
        """Delete all chunks for a given session.

        Useful for re-indexing a specific session without full re-sync.

        Args:
            session_id: Session ID to delete chunks for.

        Returns:
            Number of chunks deleted (0 if session not found).

        Raises:
            TableNotFoundError: If sessions table doesn't exist.
            DatabaseError: If delete operation fails.
        """
        if not self.table_exists():
            return 0

        try:
            table = self._connection.open_table(SESSIONS_TABLE)

            # Count before delete
            # Note: LanceDB delete doesn't return count, so we query first
            count_before = table.count_rows(f"session_id = '{session_id}'")

            if count_before > 0:
                table.delete(f"session_id = '{session_id}'")

            return int(count_before)
        except Exception as e:
            raise DatabaseError(f"Failed to delete session {session_id}: {e}") from e

    def count_chunks(self) -> int:
        """Count total chunks in the database.

        Returns:
            Total number of chunks, or 0 if table doesn't exist.
        """
        if not self.table_exists():
            return 0
        table = self._connection.open_table(SESSIONS_TABLE)
        return int(table.count_rows())

    def count_sessions(self) -> int:
        """Count unique sessions in the database.

        Returns:
            Number of unique session IDs, or 0 if table doesn't exist.
        """
        if not self.table_exists():
            return 0

        table = self._connection.open_table(SESSIONS_TABLE)
        # Use Arrow to avoid pandas dependency
        arrow_table = table.to_arrow()
        session_ids = arrow_table.column("session_id")
        unique_sessions = session_ids.unique()
        return len(unique_sessions)

    def drop_table(self) -> bool:
        """Drop the sessions table if it exists.

        Used for full re-indexing with --force flag.

        Returns:
            True if table was dropped, False if it didn't exist.

        Raises:
            DatabaseError: If drop operation fails.
        """
        if not self.table_exists():
            return False

        try:
            self._connection.drop_table(SESSIONS_TABLE)
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to drop table: {e}") from e

    def close(self) -> None:
        """Close the database connection.

        Note: LanceDB connections are lightweight and auto-close,
        but explicit close is good practice.
        """
        # LanceDB connections don't have explicit close, but we can
        # clear our reference to allow garbage collection
        object.__setattr__(self, "_connection", None)

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 20,
        repo_path: str | None = None,
    ) -> list[ChunkMatch]:
        """Search for similar chunks using ANN vector search.

        Uses LanceDB's approximate nearest neighbor search to find chunks
        with embeddings similar to the query vector. Results are ordered
        by distance (closest first) and converted to similarity scores.

        Args:
            query_vector: 768-dimensional embedding vector to search for.
            top_k: Maximum number of results to return (default: 20).
            repo_path: Optional filter to restrict search to a specific repo.
                       Uses LanceDB prefilter for efficient filtering.

        Returns:
            List of ChunkMatch objects ordered by similarity (highest first).
            Empty list if table doesn't exist or no matches found.

        Raises:
            SchemaError: If query_vector has wrong dimensions.
            DatabaseError: If search operation fails.

        Example:
            matches = db.search(query_embedding, top_k=10)
            for match in matches:
                print(f"{match.chunk.session_id}: {match.similarity:.2f}")
        """
        # Validate query vector dimensions
        if len(query_vector) != VECTOR_DIMENSIONS:
            raise SchemaError(
                f"Query vector has {len(query_vector)} dimensions, "
                f"expected {VECTOR_DIMENSIONS}"
            )

        # Return empty if table doesn't exist
        if not self.table_exists():
            return []

        try:
            table = self._connection.open_table(SESSIONS_TABLE)

            # Build search query
            # LanceDB uses L2 distance by default (lower = more similar)
            query = table.search(query_vector, vector_column_name="embedding")

            # Apply repo filter if specified (prefilter for efficiency)
            if repo_path:
                # Escape single quotes in repo_path for SQL
                escaped_path = repo_path.replace("'", "''")
                query = query.where(f"repo_path = '{escaped_path}'", prefilter=True)

            # Execute search with limit
            query = query.limit(top_k)
            results = query.to_arrow()

            # Convert Arrow table to ChunkMatch objects
            matches: list[ChunkMatch] = []
            for i in range(results.num_rows):
                # Extract fields from Arrow table
                distance = float(results.column("_distance")[i].as_py())

                # Convert L2 distance to similarity score
                # L2 distance is >= 0, lower is better
                # We use 1 / (1 + distance) to convert to 0-1 similarity
                # where higher is better
                similarity = 1.0 / (1.0 + distance)

                # Reconstruct SessionChunk from result row
                chunk = SessionChunk(
                    id=str(results.column("id")[i].as_py()),
                    session_id=str(results.column("session_id")[i].as_py()),
                    repo_path=str(results.column("repo_path")[i].as_py()),
                    chunk_index=int(results.column("chunk_index")[i].as_py()),
                    chunk_text=str(results.column("chunk_text")[i].as_py()),
                    embedding=list(results.column("embedding")[i].as_py()),
                    timestamp=int(results.column("timestamp")[i].as_py()),
                    model_used=str(results.column("model_used")[i].as_py()),
                    token_count=int(results.column("token_count")[i].as_py()),
                )

                matches.append(
                    ChunkMatch(
                        chunk=chunk,
                        similarity=similarity,
                        distance=distance,
                    )
                )

            # Results already sorted by distance (ascending) from LanceDB
            # which means highest similarity first after our conversion
            return matches

        except Exception as e:
            raise DatabaseError(f"Failed to search: {e}") from e

    def get_session_chunk_count(self, session_id: str) -> int:
        """Get the total number of chunks for a specific session.

        Useful for computing chunk_ratio in scoring algorithm
        (matching chunks / total chunks).

        Args:
            session_id: Session ID to count chunks for.

        Returns:
            Total number of chunks for the session, or 0 if not found.

        Raises:
            DatabaseError: If count operation fails.
        """
        if not self.table_exists():
            return 0

        try:
            table = self._connection.open_table(SESSIONS_TABLE)
            return int(table.count_rows(f"session_id = '{session_id}'"))
        except Exception as e:
            raise DatabaseError(
                f"Failed to count chunks for session {session_id}: {e}"
            ) from e

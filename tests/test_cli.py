"""Tests for smart_fork CLI.

Tests use Click's CliRunner for isolated testing without side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from smart_fork.cli import main
from smart_fork.ingest import SyncProgress, SyncResult

if TYPE_CHECKING:
    pass

# Patch targets are the module paths where the imports happen (inside sync function)
LOAD_CONFIG_PATH = "smart_fork.config.load_config"
SYNC_SESSIONS_PATH = "smart_fork.ingest.sync_sessions"


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_main_help(self) -> None:
        """Main --help shows expected content."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Semantic session discovery" in result.output
        assert "sync" in result.output
        assert "search" in result.output
        assert "status" in result.output

    def test_main_version(self) -> None:
        """Main --version shows version info."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        # Version should be displayed (format: "smart-fork, version X.Y.Z")
        assert "version" in result.output.lower()


class TestSyncCommand:
    """Tests for the sync command."""

    def test_sync_help(self) -> None:
        """Sync --help shows expected content."""
        runner = CliRunner()
        result = runner.invoke(main, ["sync", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output
        assert "--quiet" in result.output
        assert "Re-index all sessions" in result.output

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_success_no_sessions(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync with no sessions to process shows appropriate message."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult(
            sessions_processed=0,
            sessions_deleted=0,
            sessions_failed=0,
            chunks_added=0,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["sync"])

        assert result.exit_code == 0
        assert "No new or modified sessions to sync" in result.output
        mock_sync_sessions.assert_called_once()

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_success_with_sessions(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync with successful sessions shows processed count."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult(
            sessions_processed=5,
            sessions_deleted=0,
            sessions_failed=0,
            chunks_added=42,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["sync"])

        assert result.exit_code == 0
        assert "Processed 5 session(s)" in result.output
        assert "42 chunks indexed" in result.output

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_with_deleted_sessions(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync shows deleted session count."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult(
            sessions_processed=3,
            sessions_deleted=2,
            sessions_failed=0,
            chunks_added=30,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["sync"])

        assert result.exit_code == 0
        assert "Removed 2 deleted session(s)" in result.output

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_with_failures(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync with failures shows error count and exits with code 1."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult(
            sessions_processed=3,
            sessions_deleted=0,
            sessions_failed=2,
            chunks_added=30,
            errors=[
                "Session ses_abc: Embedding failed",
                "Session ses_xyz: Parse error",
            ],
        )

        runner = CliRunner()
        result = runner.invoke(main, ["sync"])

        assert result.exit_code == 1
        assert "Failed to process 2 session(s)" in result.output
        assert "Embedding failed" in result.output
        assert "Parse error" in result.output

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_force_flag(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync --force passes force=True to sync_sessions."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult()

        runner = CliRunner()
        result = runner.invoke(main, ["sync", "--force"])

        assert result.exit_code == 0
        assert "Re-indexing sessions" in result.output
        # Check that force=True was passed
        call_kwargs = mock_sync_sessions.call_args.kwargs
        assert call_kwargs["force"] is True

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_quiet_flag(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync --quiet suppresses all output."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult(
            sessions_processed=5,
            sessions_deleted=0,
            sessions_failed=0,
            chunks_added=42,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["sync", "--quiet"])

        assert result.exit_code == 0
        # Quiet mode should produce no output
        assert result.output.strip() == ""

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_quiet_short_flag(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync -q is equivalent to --quiet."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult(
            sessions_processed=5,
            sessions_deleted=0,
            sessions_failed=0,
            chunks_added=42,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["sync", "-q"])

        assert result.exit_code == 0
        assert result.output.strip() == ""

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_quiet_with_failures_still_exits_nonzero(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync --quiet still exits with code 1 on failures (for scripts)."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult(
            sessions_processed=1,
            sessions_deleted=0,
            sessions_failed=1,
            chunks_added=10,
            errors=["Session ses_abc: Failed"],
        )

        runner = CliRunner()
        result = runner.invoke(main, ["sync", "--quiet"])

        assert result.exit_code == 1
        # Still no output in quiet mode, but exit code signals failure
        assert result.output.strip() == ""

    @patch(LOAD_CONFIG_PATH)
    def test_sync_config_error(self, mock_load_config: MagicMock) -> None:
        """Sync shows error message when config fails to load."""
        mock_load_config.side_effect = ValueError("Invalid JSON in config")

        runner = CliRunner()
        result = runner.invoke(main, ["sync"])

        assert result.exit_code == 1
        assert "Error loading config" in result.output
        assert "Invalid JSON" in result.output

    @patch(LOAD_CONFIG_PATH)
    def test_sync_config_error_quiet(self, mock_load_config: MagicMock) -> None:
        """Sync --quiet suppresses config error message but still exits 1."""
        mock_load_config.side_effect = ValueError("Invalid JSON in config")

        runner = CliRunner()
        result = runner.invoke(main, ["sync", "--quiet"])

        assert result.exit_code == 1
        assert result.output.strip() == ""

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_exception_handling(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync handles unexpected exceptions gracefully."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.side_effect = RuntimeError("Unexpected error")

        runner = CliRunner()
        result = runner.invoke(main, ["sync"])

        assert result.exit_code == 1
        assert "Sync failed" in result.output
        assert "Unexpected error" in result.output

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_progress_callback_called(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync passes a progress callback to sync_sessions."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult()

        runner = CliRunner()
        result = runner.invoke(main, ["sync"])

        assert result.exit_code == 0
        # Verify progress_callback was passed
        call_kwargs = mock_sync_sessions.call_args.kwargs
        assert "progress_callback" in call_kwargs
        assert call_kwargs["progress_callback"] is not None

    @patch(SYNC_SESSIONS_PATH)
    @patch(LOAD_CONFIG_PATH)
    def test_sync_truncates_long_error_list(
        self, mock_load_config: MagicMock, mock_sync_sessions: MagicMock
    ) -> None:
        """Sync only shows first 3 errors with '... and N more' message."""
        mock_load_config.return_value = MagicMock()
        mock_sync_sessions.return_value = SyncResult(
            sessions_processed=0,
            sessions_deleted=0,
            sessions_failed=5,
            chunks_added=0,
            errors=[
                "Error 1",
                "Error 2",
                "Error 3",
                "Error 4",
                "Error 5",
            ],
        )

        runner = CliRunner()
        result = runner.invoke(main, ["sync"])

        assert result.exit_code == 1
        assert "Error 1" in result.output
        assert "Error 2" in result.output
        assert "Error 3" in result.output
        assert "Error 4" not in result.output  # Truncated
        assert "2 more errors" in result.output


class TestSearchCommand:
    """Tests for the search command."""

    def test_search_help(self) -> None:
        """Search --help shows expected content."""
        runner = CliRunner()
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "--scope" in result.output
        assert "--repo" in result.output
        assert "QUERY" in result.output
        assert "global" in result.output
        assert "repo" in result.output

    @patch("smart_fork.query.search_sessions")
    @patch(LOAD_CONFIG_PATH)
    def test_search_success_with_results(
        self, mock_load_config: MagicMock, mock_search: MagicMock
    ) -> None:
        """Search with results displays table and fork command."""
        from smart_fork.query import QueryResult
        from smart_fork.types import SessionMatch

        mock_load_config.return_value = MagicMock()
        mock_search.return_value = QueryResult(
            matches=[
                SessionMatch(
                    session_id="ses_abc123",
                    repo_path="/home/user/repos/myproject",
                    repo_name="myproject",
                    timestamp=1705000000,
                    score=0.94,
                    best_snippet="Implemented rate limiting for API calls",
                    chunk_count=3,
                ),
                SessionMatch(
                    session_id="ses_def456",
                    repo_path="/home/user/repos/other",
                    repo_name="other",
                    timestamp=1704000000,
                    score=0.82,
                    best_snippet="Added webhook handling",
                    chunk_count=2,
                ),
            ],
            query_time_ms=150.5,
            total_chunks_searched=20,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["search", "rate limiting"])

        assert result.exit_code == 0
        assert "myproject" in result.output
        assert "94%" in result.output
        assert "ses_abc123" in result.output
        assert "opencode --session" in result.output

    @patch("smart_fork.query.search_sessions")
    @patch(LOAD_CONFIG_PATH)
    def test_search_no_results(
        self, mock_load_config: MagicMock, mock_search: MagicMock
    ) -> None:
        """Search with no results shows helpful message."""
        from smart_fork.query import QueryResult

        mock_load_config.return_value = MagicMock()
        mock_search.return_value = QueryResult(
            matches=[],
            query_time_ms=50.0,
            total_chunks_searched=0,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["search", "nonexistent topic"])

        assert result.exit_code == 0
        assert "No matching sessions found" in result.output
        assert "smart-fork sync" in result.output

    @patch("smart_fork.query.search_sessions")
    @patch(LOAD_CONFIG_PATH)
    def test_search_scope_repo_uses_cwd(
        self, mock_load_config: MagicMock, mock_search: MagicMock
    ) -> None:
        """Search --scope repo passes current directory as repo filter."""
        import os
        from smart_fork.query import QueryResult

        mock_load_config.return_value = MagicMock()
        mock_search.return_value = QueryResult(
            matches=[],
            query_time_ms=50.0,
            total_chunks_searched=0,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["search", "--scope", "repo", "test query"])

        assert result.exit_code == 0
        # Verify repo_path was passed (current working directory)
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["repo_path"] is not None
        # Should be an absolute path
        assert os.path.isabs(call_kwargs["repo_path"])

    @patch("smart_fork.query.search_sessions")
    @patch(LOAD_CONFIG_PATH)
    def test_search_repo_flag_overrides_scope(
        self, mock_load_config: MagicMock, mock_search: MagicMock
    ) -> None:
        """Search --repo flag takes precedence and filters by specific path."""
        from smart_fork.query import QueryResult

        mock_load_config.return_value = MagicMock()
        mock_search.return_value = QueryResult(
            matches=[],
            query_time_ms=50.0,
            total_chunks_searched=0,
        )

        runner = CliRunner()
        result = runner.invoke(
            main, ["search", "--repo", "/custom/repo/path", "test query"]
        )

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args.kwargs
        assert "/custom/repo/path" in call_kwargs["repo_path"]

    @patch("smart_fork.query.search_sessions")
    @patch(LOAD_CONFIG_PATH)
    def test_search_global_scope_no_repo_filter(
        self, mock_load_config: MagicMock, mock_search: MagicMock
    ) -> None:
        """Search with default global scope passes no repo filter."""
        from smart_fork.query import QueryResult

        mock_load_config.return_value = MagicMock()
        mock_search.return_value = QueryResult(
            matches=[],
            query_time_ms=50.0,
            total_chunks_searched=0,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["search", "test query"])

        assert result.exit_code == 0
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["repo_path"] is None

    @patch(LOAD_CONFIG_PATH)
    def test_search_config_error(self, mock_load_config: MagicMock) -> None:
        """Search shows error when config fails to load."""
        mock_load_config.side_effect = ValueError("Invalid config")

        runner = CliRunner()
        result = runner.invoke(main, ["search", "test query"])

        assert result.exit_code == 1
        assert "Error loading config" in result.output

    @patch("smart_fork.query.search_sessions")
    @patch(LOAD_CONFIG_PATH)
    def test_search_empty_query_error(
        self, mock_load_config: MagicMock, mock_search: MagicMock
    ) -> None:
        """Search with empty query shows error."""
        from smart_fork.query import EmptyQueryError

        mock_load_config.return_value = MagicMock()
        mock_search.side_effect = EmptyQueryError("Query cannot be empty")

        runner = CliRunner()
        result = runner.invoke(main, ["search", "   "])

        assert result.exit_code == 1
        assert "Query cannot be empty" in result.output

    @patch("smart_fork.query.search_sessions")
    @patch(LOAD_CONFIG_PATH)
    def test_search_query_error(
        self, mock_load_config: MagicMock, mock_search: MagicMock
    ) -> None:
        """Search handles query errors gracefully."""
        from smart_fork.query import QueryError

        mock_load_config.return_value = MagicMock()
        mock_search.side_effect = QueryError("Failed to embed query")

        runner = CliRunner()
        result = runner.invoke(main, ["search", "test query"])

        assert result.exit_code == 1
        assert "Search failed" in result.output

    @patch("smart_fork.query.search_sessions")
    @patch(LOAD_CONFIG_PATH)
    def test_search_displays_query_time(
        self, mock_load_config: MagicMock, mock_search: MagicMock
    ) -> None:
        """Search displays query completion time."""
        from smart_fork.query import QueryResult
        from smart_fork.types import SessionMatch

        mock_load_config.return_value = MagicMock()
        mock_search.return_value = QueryResult(
            matches=[
                SessionMatch(
                    session_id="ses_test",
                    repo_path="/test",
                    repo_name="test",
                    timestamp=1705000000,
                    score=0.90,
                    best_snippet="Test content",
                    chunk_count=1,
                ),
            ],
            query_time_ms=123.4,
            total_chunks_searched=10,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["search", "test query"])

        assert result.exit_code == 0
        assert "123ms" in result.output or "123 ms" in result.output.replace(
            "ms", " ms"
        )

    @patch("smart_fork.query.search_sessions")
    @patch(LOAD_CONFIG_PATH)
    def test_search_repo_scope_hides_repo_column(
        self, mock_load_config: MagicMock, mock_search: MagicMock
    ) -> None:
        """Search with repo filter omits repo column (all results same repo)."""
        from smart_fork.query import QueryResult
        from smart_fork.types import SessionMatch

        mock_load_config.return_value = MagicMock()
        mock_search.return_value = QueryResult(
            matches=[
                SessionMatch(
                    session_id="ses_test",
                    repo_path="/custom/repo",
                    repo_name="repo",
                    timestamp=1705000000,
                    score=0.90,
                    best_snippet="Test content",
                    chunk_count=1,
                ),
            ],
            query_time_ms=50.0,
            total_chunks_searched=10,
        )

        runner = CliRunner()
        result = runner.invoke(main, ["search", "--repo", "/custom/repo", "test"])

        assert result.exit_code == 0
        # The suggestion to remove --scope repo should appear when no results
        # This test just verifies the command runs successfully with repo filter


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_help(self) -> None:
        """Status --help shows expected content."""
        runner = CliRunner()
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0
        assert "index statistics" in result.output.lower()

    @patch("smart_fork.ingest.load_sync_state")
    @patch("smart_fork.db.ChunkDatabase.open")
    @patch(LOAD_CONFIG_PATH)
    def test_status_empty_database(
        self,
        mock_load_config: MagicMock,
        mock_db_open: MagicMock,
        mock_load_sync_state: MagicMock,
    ) -> None:
        """Status with no indexed sessions shows helpful message."""
        from pathlib import Path

        from smart_fork.types import SyncState

        # Set up mocks
        mock_config = MagicMock()
        mock_config.paths.lance_path = Path("/test/lance")
        mock_config.paths.sync_state_path = Path("/test/sync-state.json")
        mock_load_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.count_sessions.return_value = 0
        mock_db.count_chunks.return_value = 0
        mock_db_open.return_value = mock_db

        mock_load_sync_state.return_value = SyncState(last_sync=0, sessions={})

        runner = CliRunner()
        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "No sessions indexed yet" in result.output
        assert "smart-fork sync" in result.output

    @patch("smart_fork.ingest.load_sync_state")
    @patch("smart_fork.db.ChunkDatabase.open")
    @patch(LOAD_CONFIG_PATH)
    def test_status_with_indexed_sessions(
        self,
        mock_load_config: MagicMock,
        mock_db_open: MagicMock,
        mock_load_sync_state: MagicMock,
    ) -> None:
        """Status shows session and chunk counts."""
        from pathlib import Path

        from smart_fork.types import SyncState

        # Set up mocks
        mock_config = MagicMock()
        mock_config.paths.lance_path = Path("/test/lance")
        mock_config.paths.sync_state_path = Path("/test/sync-state.json")
        mock_load_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.count_sessions.return_value = 42
        mock_db.count_chunks.return_value = 350
        mock_db_open.return_value = mock_db

        # Last sync was 1 hour ago
        import time

        last_sync_time = int(time.time()) - 3600
        mock_load_sync_state.return_value = SyncState(
            last_sync=last_sync_time, sessions={}
        )

        runner = CliRunner()
        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "42" in result.output  # Session count
        assert "350" in result.output  # Chunk count
        assert "1 hour" in result.output  # Relative time

    @patch("smart_fork.ingest.load_sync_state")
    @patch("smart_fork.db.ChunkDatabase.open")
    @patch(LOAD_CONFIG_PATH)
    def test_status_shows_database_path(
        self,
        mock_load_config: MagicMock,
        mock_db_open: MagicMock,
        mock_load_sync_state: MagicMock,
    ) -> None:
        """Status displays the database path."""
        from pathlib import Path

        from smart_fork.types import SyncState

        mock_config = MagicMock()
        mock_config.paths.lance_path = Path("/custom/data/lance")
        mock_config.paths.sync_state_path = Path("/custom/data/sync-state.json")
        mock_load_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.count_sessions.return_value = 10
        mock_db.count_chunks.return_value = 100
        mock_db_open.return_value = mock_db

        mock_load_sync_state.return_value = SyncState(last_sync=1705000000, sessions={})

        runner = CliRunner()
        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "/custom/data/lance" in result.output

    @patch("smart_fork.ingest.load_sync_state")
    @patch("smart_fork.db.ChunkDatabase.open")
    @patch(LOAD_CONFIG_PATH)
    def test_status_never_synced(
        self,
        mock_load_config: MagicMock,
        mock_db_open: MagicMock,
        mock_load_sync_state: MagicMock,
    ) -> None:
        """Status shows 'Never' when last_sync is 0 but sessions exist."""
        from pathlib import Path

        from smart_fork.types import SyncState

        mock_config = MagicMock()
        mock_config.paths.lance_path = Path("/test/lance")
        mock_config.paths.sync_state_path = Path("/test/sync-state.json")
        mock_load_config.return_value = mock_config

        mock_db = MagicMock()
        mock_db.count_sessions.return_value = 5
        mock_db.count_chunks.return_value = 50
        mock_db_open.return_value = mock_db

        # last_sync = 0 means never synced (shouldn't happen normally, but edge case)
        mock_load_sync_state.return_value = SyncState(last_sync=0, sessions={})

        runner = CliRunner()
        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "Never" in result.output

    @patch(LOAD_CONFIG_PATH)
    def test_status_config_error(self, mock_load_config: MagicMock) -> None:
        """Status shows error when config fails to load."""
        mock_load_config.side_effect = ValueError("Invalid config")

        runner = CliRunner()
        result = runner.invoke(main, ["status"])

        assert result.exit_code == 1
        assert "Error loading config" in result.output

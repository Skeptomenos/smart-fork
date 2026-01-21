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
    """Tests for the search command (stub, implemented in Task 6.2)."""

    def test_search_help(self) -> None:
        """Search --help shows expected content."""
        runner = CliRunner()
        result = runner.invoke(main, ["search", "--help"])
        assert result.exit_code == 0
        assert "--scope" in result.output
        assert "--repo" in result.output

    def test_search_stub(self) -> None:
        """Search command shows not-implemented message (to be replaced in 6.2)."""
        runner = CliRunner()
        result = runner.invoke(main, ["search", "test query"])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output


class TestStatusCommand:
    """Tests for the status command (stub, implemented in Task 6.4)."""

    def test_status_help(self) -> None:
        """Status --help shows expected content."""
        runner = CliRunner()
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0

    def test_status_stub(self) -> None:
        """Status command shows not-implemented message (to be replaced in 6.4)."""
        runner = CliRunner()
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "not yet implemented" in result.output

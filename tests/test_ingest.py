"""Tests for the ingestion pipeline.

Tests session discovery, metadata parsing, and sync diffing logic.
Uses temporary directories with mock session structures.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from smart_fork.ingest import (
    ParsedSession,
    SessionInfo,
    SessionMetadata,
    chunk_session,
    discover_sessions,
    get_sessions_to_sync,
    parse_session,
    parse_session_messages,
    parse_session_metadata,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sessions_dir(tmp_path: Path) -> Path:
    """Create a temporary sessions directory."""
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions


def create_session(
    sessions_dir: Path,
    session_id: str,
    *,
    session_data: dict[str, Any] | None = None,
    messages: list[dict[str, str]] | None = None,
    skip_session_json: bool = False,
    skip_messages_json: bool = False,
) -> Path:
    """Create a mock session directory with optional files.

    Args:
        sessions_dir: Parent directory for sessions
        session_id: Session ID (should start with ses_)
        session_data: Content for session.json
        messages: Content for messages.json
        skip_session_json: If True, don't create session.json
        skip_messages_json: If True, don't create messages.json

    Returns:
        Path to the created session directory
    """
    session_path = sessions_dir / session_id
    session_path.mkdir()

    if not skip_session_json:
        session_json = session_path / "session.json"
        # Use provided session_data if not None, otherwise use default
        data = (
            session_data
            if session_data is not None
            else {"working_directory": "/home/user/project"}
        )
        session_json.write_text(json.dumps(data))

    if not skip_messages_json:
        messages_json = session_path / "messages.json"
        msgs = messages or [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        messages_json.write_text(json.dumps(msgs))

    return session_path


# ============================================================================
# discover_sessions() Tests
# ============================================================================


class TestDiscoverSessions:
    """Tests for discover_sessions function."""

    def test_discovers_valid_sessions(self, sessions_dir: Path) -> None:
        """Should find all valid session directories."""
        create_session(sessions_dir, "ses_abc123")
        create_session(sessions_dir, "ses_def456")
        create_session(sessions_dir, "ses_ghi789")

        result = discover_sessions(sessions_dir)

        assert len(result) == 3
        session_ids = {s.session_id for s in result}
        assert session_ids == {"ses_abc123", "ses_def456", "ses_ghi789"}

    def test_returns_empty_for_nonexistent_dir(self, tmp_path: Path) -> None:
        """Should return empty list when directory doesn't exist."""
        nonexistent = tmp_path / "does_not_exist"

        result = discover_sessions(nonexistent)

        assert result == []

    def test_returns_empty_for_empty_dir(self, sessions_dir: Path) -> None:
        """Should return empty list when directory is empty."""
        result = discover_sessions(sessions_dir)

        assert result == []

    def test_skips_non_session_directories(self, sessions_dir: Path) -> None:
        """Should skip directories not starting with ses_."""
        create_session(sessions_dir, "ses_valid")
        # Create non-session directories
        (sessions_dir / "other_dir").mkdir()
        (sessions_dir / "config").mkdir()
        (sessions_dir / "backup").mkdir()

        result = discover_sessions(sessions_dir)

        assert len(result) == 1
        assert result[0].session_id == "ses_valid"

    def test_skips_files(self, sessions_dir: Path) -> None:
        """Should skip regular files in sessions directory."""
        create_session(sessions_dir, "ses_valid")
        # Create a file (not a directory)
        (sessions_dir / "ses_not_a_dir.txt").write_text("data")
        (sessions_dir / "config.json").write_text("{}")

        result = discover_sessions(sessions_dir)

        assert len(result) == 1
        assert result[0].session_id == "ses_valid"

    def test_skips_sessions_missing_session_json(self, sessions_dir: Path) -> None:
        """Should skip sessions without session.json."""
        create_session(sessions_dir, "ses_valid")
        create_session(sessions_dir, "ses_invalid", skip_session_json=True)

        result = discover_sessions(sessions_dir)

        assert len(result) == 1
        assert result[0].session_id == "ses_valid"

    def test_skips_sessions_missing_messages_json(self, sessions_dir: Path) -> None:
        """Should skip sessions without messages.json."""
        create_session(sessions_dir, "ses_valid")
        create_session(sessions_dir, "ses_invalid", skip_messages_json=True)

        result = discover_sessions(sessions_dir)

        assert len(result) == 1
        assert result[0].session_id == "ses_valid"

    def test_returns_session_info_with_correct_fields(self, sessions_dir: Path) -> None:
        """Should return SessionInfo with correct path and session_id."""
        session_path = create_session(sessions_dir, "ses_test123")

        result = discover_sessions(sessions_dir)

        assert len(result) == 1
        session = result[0]
        assert session.session_id == "ses_test123"
        assert session.path == session_path
        assert isinstance(session.last_modified, int)
        assert session.last_modified > 0

    def test_last_modified_uses_max_of_both_files(self, sessions_dir: Path) -> None:
        """Should use the maximum mtime of session.json and messages.json."""
        session_path = create_session(sessions_dir, "ses_test")

        # Modify messages.json to be newer
        time.sleep(0.1)  # Ensure time difference
        messages_json = session_path / "messages.json"
        messages_json.write_text("[]")

        result = discover_sessions(sessions_dir)

        assert len(result) == 1
        # last_modified should reflect the messages.json time
        expected_mtime = int(messages_json.stat().st_mtime)
        assert result[0].last_modified == expected_mtime

    def test_sorted_by_last_modified_descending(self, sessions_dir: Path) -> None:
        """Should return sessions sorted newest first."""
        import os

        # Create sessions
        old_path = create_session(sessions_dir, "ses_old")
        middle_path = create_session(sessions_dir, "ses_middle")
        new_path = create_session(sessions_dir, "ses_new")

        # Set explicit mtimes to ensure ordering (using utime for reliability)
        # Old: 1000 seconds ago, Middle: 500 seconds ago, New: now
        now = time.time()
        old_time = now - 1000
        middle_time = now - 500
        new_time = now

        for f in (old_path / "session.json", old_path / "messages.json"):
            os.utime(f, (old_time, old_time))
        for f in (middle_path / "session.json", middle_path / "messages.json"):
            os.utime(f, (middle_time, middle_time))
        for f in (new_path / "session.json", new_path / "messages.json"):
            os.utime(f, (new_time, new_time))

        result = discover_sessions(sessions_dir)

        assert len(result) == 3
        assert result[0].session_id == "ses_new"
        assert result[1].session_id == "ses_middle"
        assert result[2].session_id == "ses_old"

    def test_handles_path_with_tilde(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should expand ~ in path."""
        # Create sessions dir in tmp_path
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        create_session(sessions_dir, "ses_test")

        # Monkeypatch expanduser to return our tmp path
        def mock_expanduser(p: Path) -> Path:
            if str(p).startswith("~"):
                return tmp_path / str(p)[2:]  # Remove ~/
            return p

        monkeypatch.setattr(Path, "expanduser", mock_expanduser)

        # This should work with ~ expansion
        result = discover_sessions(Path("~/sessions"))

        assert len(result) == 1
        assert result[0].session_id == "ses_test"

    def test_handles_file_as_sessions_dir(self, tmp_path: Path) -> None:
        """Should return empty when sessions_dir is a file, not directory."""
        file_path = tmp_path / "not_a_dir"
        file_path.write_text("I'm a file")

        result = discover_sessions(file_path)

        assert result == []


# ============================================================================
# parse_session_metadata() Tests
# ============================================================================


class TestParseSessionMetadata:
    """Tests for parse_session_metadata function."""

    def test_parses_complete_metadata(self, sessions_dir: Path) -> None:
        """Should parse all fields when present."""
        session_data = {
            "working_directory": "/home/user/project",
            "timestamp": 1705848000,
            "parent_session_id": "ses_parent123",
            "model": "claude-3-opus",
        }
        session_path = create_session(
            sessions_dir, "ses_test", session_data=session_data
        )

        result = parse_session_metadata(session_path)

        assert result is not None
        assert result.session_id == "ses_test"
        assert result.repo_path == "/home/user/project"
        assert result.timestamp == 1705848000
        assert result.parent_session_id == "ses_parent123"
        assert result.model == "claude-3-opus"

    def test_parses_minimal_metadata(self, sessions_dir: Path) -> None:
        """Should handle minimal session.json with defaults."""
        session_data: dict[str, Any] = {}
        session_path = create_session(
            sessions_dir, "ses_minimal", session_data=session_data
        )

        result = parse_session_metadata(session_path)

        assert result is not None
        assert result.session_id == "ses_minimal"
        assert result.repo_path == ""
        assert result.parent_session_id is None
        assert result.model is None
        # timestamp should fall back to file mtime
        assert result.timestamp > 0

    def test_handles_missing_session_json(self, sessions_dir: Path) -> None:
        """Should return None when session.json doesn't exist."""
        session_path = create_session(sessions_dir, "ses_test", skip_session_json=True)

        result = parse_session_metadata(session_path)

        assert result is None

    def test_handles_invalid_json(self, sessions_dir: Path) -> None:
        """Should return None for malformed JSON."""
        session_path = sessions_dir / "ses_invalid"
        session_path.mkdir()
        (session_path / "session.json").write_text("not valid json{")
        (session_path / "messages.json").write_text("[]")

        result = parse_session_metadata(session_path)

        assert result is None

    def test_handles_non_object_json(self, sessions_dir: Path) -> None:
        """Should return None when JSON is not an object."""
        session_path = sessions_dir / "ses_array"
        session_path.mkdir()
        (session_path / "session.json").write_text('["not", "an", "object"]')
        (session_path / "messages.json").write_text("[]")

        result = parse_session_metadata(session_path)

        assert result is None

    def test_handles_alternate_field_names(self, sessions_dir: Path) -> None:
        """Should support alternate field names for repo_path."""
        # Test repo_path
        session_data = {"repo_path": "/path/from/repo_path"}
        path1 = create_session(sessions_dir, "ses_repo_path", session_data=session_data)
        result1 = parse_session_metadata(path1)
        assert result1 is not None
        assert result1.repo_path == "/path/from/repo_path"

        # Test cwd
        session_data = {"cwd": "/path/from/cwd"}
        path2 = create_session(sessions_dir, "ses_cwd", session_data=session_data)
        result2 = parse_session_metadata(path2)
        assert result2 is not None
        assert result2.repo_path == "/path/from/cwd"

        # Test workdir
        session_data = {"workdir": "/path/from/workdir"}
        path3 = create_session(sessions_dir, "ses_workdir", session_data=session_data)
        result3 = parse_session_metadata(path3)
        assert result3 is not None
        assert result3.repo_path == "/path/from/workdir"

    def test_handles_forked_from_field(self, sessions_dir: Path) -> None:
        """Should support forked_from as alternate for parent_session_id."""
        session_data = {"forked_from": "ses_original"}
        session_path = create_session(
            sessions_dir, "ses_fork", session_data=session_data
        )

        result = parse_session_metadata(session_path)

        assert result is not None
        assert result.parent_session_id == "ses_original"

    def test_handles_created_at_field(self, sessions_dir: Path) -> None:
        """Should support created_at as alternate for timestamp."""
        session_data = {"created_at": 1705900000}
        session_path = create_session(
            sessions_dir, "ses_created", session_data=session_data
        )

        result = parse_session_metadata(session_path)

        assert result is not None
        assert result.timestamp == 1705900000

    def test_handles_float_timestamp(self, sessions_dir: Path) -> None:
        """Should convert float timestamp to int."""
        session_data = {"timestamp": 1705900000.123}
        session_path = create_session(
            sessions_dir, "ses_float", session_data=session_data
        )

        result = parse_session_metadata(session_path)

        assert result is not None
        assert result.timestamp == 1705900000
        assert isinstance(result.timestamp, int)

    def test_handles_string_timestamp(self, sessions_dir: Path) -> None:
        """Should convert string timestamp to int."""
        session_data = {"timestamp": "1705900000"}
        session_path = create_session(
            sessions_dir, "ses_string_ts", session_data=session_data
        )

        result = parse_session_metadata(session_path)

        assert result is not None
        assert result.timestamp == 1705900000

    def test_working_directory_takes_precedence(self, sessions_dir: Path) -> None:
        """Should prefer working_directory over other field names."""
        session_data = {
            "working_directory": "/preferred/path",
            "repo_path": "/fallback/path",
            "cwd": "/other/path",
        }
        session_path = create_session(
            sessions_dir, "ses_precedence", session_data=session_data
        )

        result = parse_session_metadata(session_path)

        assert result is not None
        assert result.repo_path == "/preferred/path"


# ============================================================================
# get_sessions_to_sync() Tests
# ============================================================================


class TestGetSessionsToSync:
    """Tests for get_sessions_to_sync function."""

    def test_all_new_sessions(self) -> None:
        """Should return all sessions when sync state is empty."""
        sessions = [
            SessionInfo("ses_a", Path("/sessions/ses_a"), 1000),
            SessionInfo("ses_b", Path("/sessions/ses_b"), 2000),
        ]
        sync_state: dict[str, int] = {}

        to_process, to_delete = get_sessions_to_sync(sessions, sync_state)

        assert len(to_process) == 2
        assert to_delete == []

    def test_no_changes(self) -> None:
        """Should return empty when all sessions unchanged."""
        sessions = [
            SessionInfo("ses_a", Path("/sessions/ses_a"), 1000),
            SessionInfo("ses_b", Path("/sessions/ses_b"), 2000),
        ]
        sync_state = {
            "ses_a": 1000,  # Same timestamp
            "ses_b": 2000,  # Same timestamp
        }

        to_process, to_delete = get_sessions_to_sync(sessions, sync_state)

        assert len(to_process) == 0
        assert to_delete == []

    def test_modified_sessions(self) -> None:
        """Should return sessions with newer timestamps."""
        sessions = [
            SessionInfo("ses_a", Path("/sessions/ses_a"), 1000),
            SessionInfo("ses_b", Path("/sessions/ses_b"), 2500),  # Modified
        ]
        sync_state = {
            "ses_a": 1000,  # Unchanged
            "ses_b": 2000,  # Old timestamp
        }

        to_process, to_delete = get_sessions_to_sync(sessions, sync_state)

        assert len(to_process) == 1
        assert to_process[0].session_id == "ses_b"
        assert to_delete == []

    def test_deleted_sessions(self) -> None:
        """Should identify sessions in sync state but not on disk."""
        sessions = [
            SessionInfo("ses_a", Path("/sessions/ses_a"), 1000),
        ]
        sync_state = {
            "ses_a": 1000,
            "ses_deleted": 500,  # No longer on disk
        }

        to_process, to_delete = get_sessions_to_sync(sessions, sync_state)

        assert len(to_process) == 0
        assert to_delete == ["ses_deleted"]

    def test_mixed_new_modified_deleted(self) -> None:
        """Should handle mix of new, modified, unchanged, and deleted."""
        sessions = [
            SessionInfo("ses_new", Path("/sessions/ses_new"), 3000),  # New
            SessionInfo(
                "ses_modified", Path("/sessions/ses_modified"), 2500
            ),  # Modified
            SessionInfo(
                "ses_unchanged", Path("/sessions/ses_unchanged"), 1000
            ),  # Unchanged
        ]
        sync_state = {
            "ses_modified": 2000,  # Old timestamp
            "ses_unchanged": 1000,  # Same timestamp
            "ses_deleted": 500,  # No longer exists
        }

        to_process, to_delete = get_sessions_to_sync(sessions, sync_state)

        process_ids = {s.session_id for s in to_process}
        assert process_ids == {"ses_new", "ses_modified"}
        assert to_delete == ["ses_deleted"]

    def test_force_sync_returns_all(self) -> None:
        """Should return all sessions when force=True."""
        sessions = [
            SessionInfo("ses_a", Path("/sessions/ses_a"), 1000),
            SessionInfo("ses_b", Path("/sessions/ses_b"), 2000),
        ]
        sync_state = {
            "ses_a": 1000,  # Would be skipped normally
            "ses_b": 2000,  # Would be skipped normally
        }

        to_process, to_delete = get_sessions_to_sync(sessions, sync_state, force=True)

        assert len(to_process) == 2
        # Force sync doesn't delete anything
        assert to_delete == []

    def test_force_sync_empty_state(self) -> None:
        """Force sync works with empty sync state."""
        sessions = [
            SessionInfo("ses_a", Path("/sessions/ses_a"), 1000),
        ]

        to_process, to_delete = get_sessions_to_sync(sessions, {}, force=True)

        assert len(to_process) == 1
        assert to_delete == []

    def test_empty_sessions_list(self) -> None:
        """Should handle empty sessions list."""
        sync_state = {"ses_old": 1000}

        to_process, to_delete = get_sessions_to_sync([], sync_state)

        assert to_process == []
        assert to_delete == ["ses_old"]


# ============================================================================
# SessionInfo and SessionMetadata Dataclass Tests
# ============================================================================


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_creation(self) -> None:
        """Should create SessionInfo with all fields."""
        info = SessionInfo(
            session_id="ses_test",
            path=Path("/sessions/ses_test"),
            last_modified=1705848000,
        )

        assert info.session_id == "ses_test"
        assert info.path == Path("/sessions/ses_test")
        assert info.last_modified == 1705848000

    def test_equality(self) -> None:
        """Should compare equal with same values."""
        info1 = SessionInfo("ses_a", Path("/a"), 1000)
        info2 = SessionInfo("ses_a", Path("/a"), 1000)

        assert info1 == info2


class TestSessionMetadata:
    """Tests for SessionMetadata dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Should create with optional fields as None."""
        meta = SessionMetadata(
            session_id="ses_test",
            repo_path="/project",
            timestamp=1705848000,
        )

        assert meta.session_id == "ses_test"
        assert meta.repo_path == "/project"
        assert meta.timestamp == 1705848000
        assert meta.parent_session_id is None
        assert meta.model is None

    def test_creation_with_all_fields(self) -> None:
        """Should create with all optional fields."""
        meta = SessionMetadata(
            session_id="ses_test",
            repo_path="/project",
            timestamp=1705848000,
            parent_session_id="ses_parent",
            model="claude-3-opus",
        )

        assert meta.parent_session_id == "ses_parent"
        assert meta.model == "claude-3-opus"


# ============================================================================
# parse_session_messages() Tests
# ============================================================================


class TestParseSessionMessages:
    """Tests for parse_session_messages function."""

    def test_parses_valid_messages(self, sessions_dir: Path) -> None:
        """Should parse array of message objects."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing great!"},
            {"role": "user", "content": "That's good to hear."},
        ]
        session_path = create_session(sessions_dir, "ses_test", messages=messages)

        result = parse_session_messages(session_path)

        assert len(result) == 3
        assert result[0] == {"role": "user", "content": "Hello, how are you?"}
        assert result[1] == {"role": "assistant", "content": "I'm doing great!"}
        assert result[2] == {"role": "user", "content": "That's good to hear."}

    def test_returns_empty_for_missing_file(self, sessions_dir: Path) -> None:
        """Should return empty list when messages.json doesn't exist."""
        session_path = create_session(sessions_dir, "ses_test", skip_messages_json=True)

        result = parse_session_messages(session_path)

        assert result == []

    def test_returns_empty_for_invalid_json(self, sessions_dir: Path) -> None:
        """Should return empty list for malformed JSON."""
        session_path = sessions_dir / "ses_invalid"
        session_path.mkdir()
        (session_path / "session.json").write_text("{}")
        (session_path / "messages.json").write_text("not valid json[")

        result = parse_session_messages(session_path)

        assert result == []

    def test_returns_empty_for_non_array_json(self, sessions_dir: Path) -> None:
        """Should return empty list when JSON is not an array."""
        session_path = sessions_dir / "ses_object"
        session_path.mkdir()
        (session_path / "session.json").write_text("{}")
        (session_path / "messages.json").write_text('{"not": "an array"}')

        result = parse_session_messages(session_path)

        assert result == []

    def test_skips_messages_with_empty_content(self, sessions_dir: Path) -> None:
        """Should skip messages without content."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},  # Empty content
            {"role": "user", "content": "Still here"},
        ]
        session_path = create_session(sessions_dir, "ses_empty", messages=messages)

        result = parse_session_messages(session_path)

        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Still here"

    def test_skips_messages_without_content_key(self, sessions_dir: Path) -> None:
        """Should skip messages that don't have content key."""
        session_path = sessions_dir / "ses_no_content"
        session_path.mkdir()
        (session_path / "session.json").write_text("{}")
        # Custom messages with missing content key
        messages_data = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant"},  # No content key
            {"role": "user", "content": "World"},
        ]
        (session_path / "messages.json").write_text(json.dumps(messages_data))

        result = parse_session_messages(session_path)

        assert len(result) == 2

    def test_defaults_missing_role_to_unknown(self, sessions_dir: Path) -> None:
        """Should use 'unknown' for messages without role."""
        session_path = sessions_dir / "ses_no_role"
        session_path.mkdir()
        (session_path / "session.json").write_text("{}")
        messages_data = [{"content": "I have no role"}]
        (session_path / "messages.json").write_text(json.dumps(messages_data))

        result = parse_session_messages(session_path)

        assert len(result) == 1
        assert result[0]["role"] == "unknown"
        assert result[0]["content"] == "I have no role"

    def test_skips_non_object_messages(self, sessions_dir: Path) -> None:
        """Should skip array elements that aren't objects."""
        session_path = sessions_dir / "ses_mixed"
        session_path.mkdir()
        (session_path / "session.json").write_text("{}")
        messages_data = [
            {"role": "user", "content": "Valid message"},
            "just a string",
            123,
            None,
            {"role": "assistant", "content": "Another valid one"},
        ]
        (session_path / "messages.json").write_text(json.dumps(messages_data))

        result = parse_session_messages(session_path)

        assert len(result) == 2
        assert result[0]["content"] == "Valid message"
        assert result[1]["content"] == "Another valid one"

    def test_returns_empty_for_empty_array(self, sessions_dir: Path) -> None:
        """Should return empty list for empty message array."""
        # Create session directory manually with empty messages array
        session_path = sessions_dir / "ses_empty_msgs"
        session_path.mkdir()
        (session_path / "session.json").write_text("{}")
        (session_path / "messages.json").write_text("[]")

        result = parse_session_messages(session_path)

        assert result == []

    def test_converts_non_string_values_to_strings(self, sessions_dir: Path) -> None:
        """Should convert role and content to strings."""
        session_path = sessions_dir / "ses_types"
        session_path.mkdir()
        (session_path / "session.json").write_text("{}")
        # Non-string values that should be converted
        messages_data = [{"role": 123, "content": "message with int role"}]
        (session_path / "messages.json").write_text(json.dumps(messages_data))

        result = parse_session_messages(session_path)

        assert len(result) == 1
        assert result[0]["role"] == "123"
        assert result[0]["content"] == "message with int role"


# ============================================================================
# parse_session() Tests
# ============================================================================


class TestParseSession:
    """Tests for parse_session function."""

    def test_parses_complete_session(self, sessions_dir: Path) -> None:
        """Should parse both metadata and messages."""
        session_data = {
            "working_directory": "/home/user/project",
            "timestamp": 1705848000,
            "parent_session_id": "ses_parent",
        }
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        session_path = create_session(
            sessions_dir, "ses_full", session_data=session_data, messages=messages
        )

        result = parse_session(session_path)

        assert result is not None
        assert result.session_id == "ses_full"
        assert result.repo_path == "/home/user/project"
        assert result.timestamp == 1705848000
        assert result.parent_session_id == "ses_parent"
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "user"
        assert result.messages[1]["role"] == "assistant"

    def test_returns_none_for_missing_metadata(self, sessions_dir: Path) -> None:
        """Should return None when session.json is missing."""
        session_path = create_session(
            sessions_dir, "ses_no_meta", skip_session_json=True
        )

        result = parse_session(session_path)

        assert result is None

    def test_returns_session_with_empty_messages(self, sessions_dir: Path) -> None:
        """Should return session even when messages.json is missing."""
        session_path = create_session(
            sessions_dir, "ses_no_msgs", skip_messages_json=True
        )
        # Need to create a valid session.json since we're not skipping it
        (session_path / "session.json").write_text('{"working_directory": "/test"}')

        # But we need messages.json for discover_sessions to find it
        # So let's create it then parse
        # Actually, parse_session doesn't require messages.json to exist
        # Let's directly test the function

        # Create a session directory manually
        session_path_manual = sessions_dir / "ses_manual"
        session_path_manual.mkdir()
        (session_path_manual / "session.json").write_text(
            '{"working_directory": "/project"}'
        )
        # No messages.json

        result = parse_session(session_path_manual)

        assert result is not None
        assert result.session_id == "ses_manual"
        assert result.messages == []

    def test_handles_invalid_messages(self, sessions_dir: Path) -> None:
        """Should handle invalid messages.json gracefully."""
        session_path = sessions_dir / "ses_bad_msgs"
        session_path.mkdir()
        (session_path / "session.json").write_text('{"working_directory": "/test"}')
        (session_path / "messages.json").write_text("not valid json")

        result = parse_session(session_path)

        assert result is not None
        assert result.messages == []


# ============================================================================
# chunk_session() Tests
# ============================================================================


class TestChunkSession:
    """Tests for chunk_session function."""

    def test_chunks_simple_session(self, sessions_dir: Path) -> None:
        """Should create SessionChunk objects from parsed session."""
        session = ParsedSession(
            session_id="ses_test123",
            repo_path="/home/user/project",
            timestamp=1705848000,
            parent_session_id=None,
            messages=[
                {"role": "user", "content": "Hello, I need help with Python."},
                {"role": "assistant", "content": "Of course! What do you need?"},
            ],
        )

        result = chunk_session(session, model_used="text-embedding-004")

        # Should have at least one chunk
        assert len(result) >= 1
        # Verify chunk structure
        chunk = result[0]
        assert chunk.id == "ses_test123_chunk_0"
        assert chunk.session_id == "ses_test123"
        assert chunk.repo_path == "/home/user/project"
        assert chunk.chunk_index == 0
        assert chunk.timestamp == 1705848000
        assert chunk.model_used == "text-embedding-004"
        assert chunk.embedding == []  # Not filled yet
        assert chunk.token_count > 0
        assert "Hello" in chunk.chunk_text

    def test_returns_empty_for_empty_messages(self) -> None:
        """Should return empty list when session has no messages."""
        session = ParsedSession(
            session_id="ses_empty",
            repo_path="/test",
            timestamp=1000,
            parent_session_id=None,
            messages=[],
        )

        result = chunk_session(session, model_used="test-model")

        assert result == []

    def test_chunk_ids_are_sequential(self) -> None:
        """Should create sequential chunk IDs."""
        # Create a session with enough content to produce multiple chunks
        long_content = "This is a test. " * 1000  # ~4000+ tokens
        session = ParsedSession(
            session_id="ses_multi",
            repo_path="/test",
            timestamp=1000,
            parent_session_id=None,
            messages=[{"role": "user", "content": long_content}],
        )

        result = chunk_session(session, model_used="test-model")

        # Should have multiple chunks
        assert len(result) >= 2
        # Verify IDs are sequential
        for i, chunk in enumerate(result):
            assert chunk.id == f"ses_multi_chunk_{i}"
            assert chunk.chunk_index == i

    def test_preserves_parent_session_id_not_in_chunk(self) -> None:
        """Should not include parent_session_id in chunk (it's session-level)."""
        session = ParsedSession(
            session_id="ses_child",
            repo_path="/test",
            timestamp=1000,
            parent_session_id="ses_parent",
            messages=[{"role": "user", "content": "Test message"}],
        )

        result = chunk_session(session, model_used="test-model")

        # SessionChunk doesn't have parent_session_id field
        # It's used at query time for scoring, not stored per-chunk
        assert len(result) >= 1
        # Verify chunk has expected fields (no parent_session_id)
        chunk = result[0]
        assert hasattr(chunk, "session_id")
        assert not hasattr(chunk, "parent_session_id")

    def test_chunk_text_contains_formatted_messages(self) -> None:
        """Should format messages as 'role: content' in chunk text."""
        session = ParsedSession(
            session_id="ses_format",
            repo_path="/test",
            timestamp=1000,
            parent_session_id=None,
            messages=[
                {"role": "user", "content": "Hello world"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

        result = chunk_session(session, model_used="test-model")

        assert len(result) >= 1
        chunk_text = result[0].chunk_text
        # Should contain formatted messages
        assert "user: Hello world" in chunk_text
        assert "assistant: Hi there" in chunk_text

    def test_handles_messages_with_special_characters(self) -> None:
        """Should handle messages with special characters."""
        session = ParsedSession(
            session_id="ses_special",
            repo_path="/test",
            timestamp=1000,
            parent_session_id=None,
            messages=[
                {"role": "user", "content": "Code: `print('hello')`"},
                {"role": "assistant", "content": "Here's a Unicode: 日本語"},
            ],
        )

        result = chunk_session(session, model_used="test-model")

        assert len(result) >= 1
        chunk_text = result[0].chunk_text
        assert "print('hello')" in chunk_text
        assert "日本語" in chunk_text


# ============================================================================
# ParsedSession Dataclass Tests
# ============================================================================


class TestParsedSession:
    """Tests for ParsedSession dataclass."""

    def test_creation(self) -> None:
        """Should create ParsedSession with all fields."""
        session = ParsedSession(
            session_id="ses_test",
            repo_path="/project",
            timestamp=1705848000,
            parent_session_id="ses_parent",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert session.session_id == "ses_test"
        assert session.repo_path == "/project"
        assert session.timestamp == 1705848000
        assert session.parent_session_id == "ses_parent"
        assert len(session.messages) == 1

    def test_equality(self) -> None:
        """Should compare equal with same values."""
        msgs = [{"role": "user", "content": "Hi"}]
        session1 = ParsedSession("ses_a", "/a", 1000, None, msgs)
        session2 = ParsedSession("ses_a", "/a", 1000, None, msgs)

        assert session1 == session2

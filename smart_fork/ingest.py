"""Ingestion pipeline for Smart Fork.

Handles discovery, parsing, and indexing of OpenCode session transcripts.
Sessions are read from ~/.local/share/opencode/sessions/ (read-only).

The ingestion pipeline:
1. Discover sessions: Find all session directories in the sessions directory
2. Parse sessions: Extract metadata and messages from session files
3. Chunk sessions: Split transcript into semantic chunks for embedding
4. Embed and store: Generate embeddings and store in LanceDB

Why session discovery is a separate concern:
- Sessions may be added/removed while the pipeline runs
- Session format may evolve; isolation helps compatibility
- Testing is simpler with clear boundaries
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

from smart_fork.config import SmartForkConfig


logger = structlog.get_logger(__name__)


@dataclass
class SessionInfo:
    """Minimal info about a discovered session.

    Represents a session directory found on disk before full parsing.
    Used to track which sessions need processing during sync.

    Attributes:
        session_id: OpenCode session ID (directory name, e.g., "ses_abc123def456")
        path: Full path to the session directory
        last_modified: Unix timestamp of the most recent file modification
    """

    session_id: str
    path: Path
    last_modified: int


@dataclass
class SessionMetadata:
    """Metadata parsed from session.json.

    Contains information about the session context needed for indexing.

    Attributes:
        session_id: OpenCode session ID
        repo_path: Absolute path to repository where session occurred
        timestamp: Unix timestamp of session creation
        parent_session_id: ID of parent session if this was a fork (for chain quality scoring)
        model: Model used in the session (for reference only)
    """

    session_id: str
    repo_path: str
    timestamp: int
    parent_session_id: str | None = None
    model: str | None = None


def discover_sessions(
    sessions_dir: Path,
) -> list[SessionInfo]:
    """Find all session directories in the OpenCode sessions directory.

    Scans the sessions directory for subdirectories matching the session
    naming pattern (ses_*). Each session directory should contain:
    - session.json: Session metadata (repo_path, model, parent_id)
    - messages.json: Array of message objects {role, content}

    Sessions without required files are logged as warnings but not included
    in results to allow partial processing when some sessions are malformed.

    Args:
        sessions_dir: Path to OpenCode sessions directory
                     (typically ~/.local/share/opencode/sessions/)

    Returns:
        List of SessionInfo for all valid session directories found,
        sorted by last_modified descending (newest first).

    Note:
        This function is read-only and does not modify any files.
        It handles missing or inaccessible directories gracefully.
    """
    sessions_dir = sessions_dir.expanduser()

    if not sessions_dir.exists():
        logger.info("sessions_dir_not_found", path=str(sessions_dir))
        return []

    if not sessions_dir.is_dir():
        logger.warning("sessions_path_not_directory", path=str(sessions_dir))
        return []

    sessions: list[SessionInfo] = []

    # Iterate through session directories
    # Session directories follow the pattern: ses_<id>
    try:
        for entry in sessions_dir.iterdir():
            if not entry.is_dir():
                continue

            # Session directories start with "ses_"
            if not entry.name.startswith("ses_"):
                logger.debug("skipping_non_session_dir", name=entry.name)
                continue

            session_id = entry.name

            # Check for required files
            session_json = entry / "session.json"
            messages_json = entry / "messages.json"

            if not session_json.exists():
                logger.warning(
                    "session_missing_metadata",
                    session_id=session_id,
                    missing_file="session.json",
                )
                continue

            if not messages_json.exists():
                logger.warning(
                    "session_missing_messages",
                    session_id=session_id,
                    missing_file="messages.json",
                )
                continue

            # Determine last modified time as the max of both files
            # This ensures we detect changes to either metadata or messages
            try:
                session_mtime = int(session_json.stat().st_mtime)
                messages_mtime = int(messages_json.stat().st_mtime)
                last_modified = max(session_mtime, messages_mtime)
            except OSError as e:
                logger.warning(
                    "session_stat_failed",
                    session_id=session_id,
                    error=str(e),
                )
                continue

            sessions.append(
                SessionInfo(
                    session_id=session_id,
                    path=entry,
                    last_modified=last_modified,
                )
            )

    except PermissionError as e:
        logger.error(
            "sessions_dir_permission_denied", path=str(sessions_dir), error=str(e)
        )
        return []
    except OSError as e:
        logger.error("sessions_dir_read_error", path=str(sessions_dir), error=str(e))
        return []

    # Sort by last_modified descending (newest first)
    # This prioritizes recent sessions for incremental sync
    sessions.sort(key=lambda s: s.last_modified, reverse=True)

    logger.info(
        "sessions_discovered",
        count=len(sessions),
        sessions_dir=str(sessions_dir),
    )

    return sessions


def parse_session_metadata(session_path: Path) -> SessionMetadata | None:
    """Parse metadata from a session's session.json file.

    Extracts key metadata needed for indexing: repo path (for scoping),
    timestamp (for recency scoring), and parent session (for chain quality).

    Args:
        session_path: Path to the session directory (not the JSON file)

    Returns:
        SessionMetadata if parsing succeeds, None if file is missing or invalid.

    Note:
        Defensively handles missing or malformed fields. The only required
        field is session_id (derived from directory name). Other fields
        fall back to sensible defaults when missing.
    """
    session_json = session_path / "session.json"
    session_id = session_path.name

    if not session_json.exists():
        logger.warning(
            "session_json_not_found",
            session_id=session_id,
            path=str(session_json),
        )
        return None

    try:
        with open(session_json, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(
            "session_json_invalid",
            session_id=session_id,
            error=str(e),
        )
        return None
    except OSError as e:
        logger.warning(
            "session_json_read_error",
            session_id=session_id,
            error=str(e),
        )
        return None

    if not isinstance(data, dict):
        logger.warning(
            "session_json_not_object",
            session_id=session_id,
            actual_type=type(data).__name__,
        )
        return None

    # Extract repo_path - try multiple possible field names
    # OpenCode may use "working_directory", "repo_path", "cwd", etc.
    repo_path = (
        data.get("working_directory")
        or data.get("repo_path")
        or data.get("cwd")
        or data.get("workdir")
        or ""
    )

    # Extract timestamp - use file mtime as fallback
    timestamp = data.get("timestamp") or data.get("created_at")
    if timestamp is None:
        try:
            timestamp = int(session_json.stat().st_mtime)
        except OSError:
            timestamp = 0

    # Ensure timestamp is int
    if isinstance(timestamp, float):
        timestamp = int(timestamp)
    elif isinstance(timestamp, str):
        try:
            timestamp = int(float(timestamp))
        except ValueError:
            timestamp = 0

    # Extract parent session ID for chain quality scoring
    parent_session_id = data.get("parent_session_id") or data.get("forked_from")

    # Extract model for reference
    model = data.get("model")

    return SessionMetadata(
        session_id=session_id,
        repo_path=str(repo_path),
        timestamp=timestamp,
        parent_session_id=parent_session_id,
        model=model,
    )


def get_sessions_to_sync(
    sessions: list[SessionInfo],
    sync_state_sessions: dict[str, int],
    force: bool = False,
) -> tuple[list[SessionInfo], list[str]]:
    """Determine which sessions need to be synced.

    Compares discovered sessions against sync state to find:
    - New sessions: Not in sync state
    - Modified sessions: last_modified > sync state timestamp
    - Deleted sessions: In sync state but not discovered

    Args:
        sessions: List of discovered sessions from discover_sessions()
        sync_state_sessions: Map of session_id -> last_sync_timestamp from SyncState
        force: If True, return all sessions regardless of sync state

    Returns:
        Tuple of (sessions_to_process, session_ids_to_delete):
        - sessions_to_process: SessionInfo list of new/modified sessions
        - session_ids_to_delete: List of session IDs to remove from index
    """
    if force:
        # Force sync: process all discovered sessions, delete nothing
        logger.info("force_sync_requested", session_count=len(sessions))
        return sessions, []

    sessions_to_process: list[SessionInfo] = []
    discovered_ids = {s.session_id for s in sessions}

    for session in sessions:
        last_synced = sync_state_sessions.get(session.session_id)

        if last_synced is None:
            # New session - never synced
            sessions_to_process.append(session)
            logger.debug("new_session", session_id=session.session_id)
        elif session.last_modified > last_synced:
            # Modified session - needs re-sync
            sessions_to_process.append(session)
            logger.debug(
                "modified_session",
                session_id=session.session_id,
                last_modified=session.last_modified,
                last_synced=last_synced,
            )

    # Find deleted sessions (in sync state but not on disk)
    session_ids_to_delete = [
        sid for sid in sync_state_sessions if sid not in discovered_ids
    ]

    if session_ids_to_delete:
        logger.info("deleted_sessions", count=len(session_ids_to_delete))

    logger.info(
        "sync_diff_computed",
        new_or_modified=len(sessions_to_process),
        deleted=len(session_ids_to_delete),
        unchanged=len(sessions) - len(sessions_to_process),
    )

    return sessions_to_process, session_ids_to_delete

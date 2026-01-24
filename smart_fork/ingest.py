"""Ingestion pipeline for Smart Fork.

Handles discovery, parsing, and indexing of OpenCode session transcripts.

OpenCode Storage Structure (actual format as of 2026-01):
  ~/.local/share/opencode/storage/
    session/<project_hash>/ses_*.json  (session metadata)
    message/ses_*/msg_*.json           (message metadata)
    part/msg_*/prt_*.json              (message content with type=="text")

Legacy Format (for testing):
  sessions/ses_*/session.json + messages.json

The ingestion pipeline:
1. Discover sessions: Find all session files across project hash directories
2. Parse sessions: Extract metadata and collect messages from part files
3. Chunk sessions: Split transcript into semantic chunks for embedding
4. Embed and store: Generate embeddings and store in LanceDB

Why session discovery is a separate concern:
- Sessions may be added/removed while the pipeline runs
- Session format may evolve; isolation helps compatibility
- Testing is simpler with clear boundaries
"""

from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Generator

import structlog

from smart_fork.chunker import Chunk, Chunker, chunk_messages
from smart_fork.config import ChunkingConfig, SmartForkConfig, load_config
from smart_fork.db import ChunkDatabase, DatabaseError
from smart_fork.embeddings import EmbeddingError, EmbeddingProvider, create_provider
from smart_fork.types import SessionChunk, SyncState


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


def discover_sessions_from_storage(
    storage_dir: Path,
) -> list[SessionInfo]:
    """Find all sessions in the OpenCode storage directory (actual format).

    Scans storage/session/<project_hash>/ses_*.json for session metadata files.
    This is the actual OpenCode storage format as of 2026-01.

    Session metadata format (ses_*.json):
        {
            "id": "ses_abc123...",
            "directory": "/path/to/repo",  # repo_path
            "time": {"created": 1234567890123, "updated": ...}  # milliseconds
        }

    Args:
        storage_dir: Path to OpenCode storage directory
                    (typically ~/.local/share/opencode/storage/)

    Returns:
        List of SessionInfo for all valid sessions found,
        sorted by last_modified descending (newest first).
    """
    storage_dir = storage_dir.expanduser()
    session_metadata_dir = storage_dir / "session"

    if not session_metadata_dir.exists():
        logger.info("session_metadata_dir_not_found", path=str(session_metadata_dir))
        return []

    if not session_metadata_dir.is_dir():
        logger.warning(
            "session_metadata_path_not_directory", path=str(session_metadata_dir)
        )
        return []

    sessions: list[SessionInfo] = []

    try:
        # Iterate through project hash directories
        for project_dir in session_metadata_dir.iterdir():
            if not project_dir.is_dir():
                continue

            # Iterate through session JSON files in each project
            for session_file in project_dir.glob("ses_*.json"):
                try:
                    session_id = session_file.stem  # e.g., "ses_abc123..."

                    # Get last modified time from file
                    last_modified = int(session_file.stat().st_mtime)

                    # Store the file path (not directory) for the new format
                    sessions.append(
                        SessionInfo(
                            session_id=session_id,
                            path=session_file,  # File path, not directory
                            last_modified=last_modified,
                        )
                    )

                except OSError as e:
                    logger.warning(
                        "session_file_stat_failed",
                        session_file=str(session_file),
                        error=str(e),
                    )
                    continue

    except PermissionError as e:
        logger.error(
            "session_metadata_dir_permission_denied",
            path=str(session_metadata_dir),
            error=str(e),
        )
        return []
    except OSError as e:
        logger.error(
            "session_metadata_dir_read_error",
            path=str(session_metadata_dir),
            error=str(e),
        )
        return []

    # Sort by last_modified descending (newest first)
    sessions.sort(key=lambda s: s.last_modified, reverse=True)

    logger.info(
        "sessions_discovered_from_storage",
        count=len(sessions),
        storage_dir=str(storage_dir),
    )

    return sessions


def parse_session_metadata_from_file(session_file: Path) -> SessionMetadata | None:
    """Parse metadata from a session JSON file (actual OpenCode format).

    Actual format (ses_*.json):
        {
            "id": "ses_abc123...",
            "directory": "/path/to/repo",
            "time": {"created": 1234567890123, "updated": ...}  # milliseconds
        }

    Args:
        session_file: Path to the session JSON file (not directory)

    Returns:
        SessionMetadata if parsing succeeds, None if file is missing or invalid.
    """
    if not session_file.exists():
        logger.warning("session_file_not_found", path=str(session_file))
        return None

    try:
        with open(session_file, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(
            "session_file_invalid_json",
            path=str(session_file),
            error=str(e),
        )
        return None
    except OSError as e:
        logger.warning(
            "session_file_read_error",
            path=str(session_file),
            error=str(e),
        )
        return None

    if not isinstance(data, dict):
        logger.warning(
            "session_file_not_object",
            path=str(session_file),
            actual_type=type(data).__name__,
        )
        return None

    # Extract session_id from file or data
    session_id = data.get("id") or session_file.stem

    # Extract repo_path from "directory" field
    repo_path = data.get("directory") or ""

    # Extract timestamp from time.created (milliseconds -> seconds)
    # Fall back to file mtime if not present or zero
    time_data = data.get("time", {})
    timestamp_ms = time_data.get("created")
    if isinstance(timestamp_ms, int) and timestamp_ms > 0:
        timestamp = timestamp_ms // 1000
    elif isinstance(timestamp_ms, float) and timestamp_ms > 0:
        timestamp = int(timestamp_ms // 1000)
    else:
        # Fall back to file modification time
        try:
            timestamp = int(session_file.stat().st_mtime)
        except OSError:
            timestamp = 0

    # Extract parent_session_id from "parentID" field (actual OpenCode format)
    # This enables chain_quality scoring for forked sessions
    parent_id = data.get("parentID")
    parent_session_id = str(parent_id) if parent_id else None

    # Model is not stored in session metadata in actual format
    model = data.get("model")

    return SessionMetadata(
        session_id=str(session_id),
        repo_path=str(repo_path),
        timestamp=timestamp,
        parent_session_id=parent_session_id,
        model=model,
    )


def parse_session_messages_from_storage(
    session_id: str,
    storage_dir: Path,
) -> list[dict[str, str]]:
    """Parse messages from OpenCode storage (actual format).

    Collects messages from:
    - storage/message/<session_id>/msg_*.json (message metadata)
    - storage/part/<msg_id>/prt_*.json (message content where type=="text")

    Args:
        session_id: The session ID (e.g., "ses_abc123...")
        storage_dir: Path to OpenCode storage directory

    Returns:
        List of message dicts with 'role' and 'content' keys,
        sorted by creation time.
    """
    storage_dir = storage_dir.expanduser()
    message_dir = storage_dir / "message" / session_id

    if not message_dir.exists():
        logger.debug(
            "session_message_dir_not_found",
            session_id=session_id,
            path=str(message_dir),
        )
        return []

    # Collect message files and sort by creation time
    message_files: list[tuple[int, Path]] = []
    for msg_file in message_dir.glob("msg_*.json"):
        try:
            with open(msg_file, encoding="utf-8") as f:
                msg_data = json.load(f)
            created_time = msg_data.get("time", {}).get("created", 0)
            message_files.append((created_time, msg_file))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(
                "message_file_read_error",
                path=str(msg_file),
                error=str(e),
            )
            continue

    # Sort by creation time
    message_files.sort(key=lambda x: x[0])

    # Collect messages with content from parts
    messages: list[dict[str, str]] = []
    part_dir = storage_dir / "part"

    for _, msg_file in message_files:
        try:
            with open(msg_file, encoding="utf-8") as f:
                msg_data = json.load(f)

            msg_id = msg_data.get("id", "")
            role = msg_data.get("role", "unknown")

            # Collect text content from parts
            msg_part_dir = part_dir / msg_id
            if not msg_part_dir.exists():
                # Try using the message directory within the part folder
                # Some versions might store parts differently
                continue

            # Collect text parts sorted by start time
            text_parts: list[tuple[int, str]] = []
            for part_file in msg_part_dir.glob("prt_*.json"):
                try:
                    with open(part_file, encoding="utf-8") as f:
                        part_data = json.load(f)

                    # Only include text type parts
                    if part_data.get("type") != "text":
                        continue

                    text = part_data.get("text", "")
                    if not text:
                        continue

                    start_time = part_data.get("time", {}).get("start", 0)
                    text_parts.append((start_time, text))

                except (json.JSONDecodeError, OSError):
                    continue

            if not text_parts:
                continue

            # Sort by start time and concatenate
            text_parts.sort(key=lambda x: x[0])
            content = "\n".join(part[1] for part in text_parts)

            if content.strip():
                messages.append({"role": str(role), "content": content})

        except (json.JSONDecodeError, OSError) as e:
            logger.debug(
                "message_parse_error",
                path=str(msg_file),
                error=str(e),
            )
            continue

    logger.debug(
        "messages_parsed_from_storage",
        session_id=session_id,
        message_count=len(messages),
    )

    return messages


def parse_session_from_storage(
    session_file: Path,
    storage_dir: Path,
) -> ParsedSession | None:
    """Parse a complete session from OpenCode storage format.

    Combines parse_session_metadata_from_file and parse_session_messages_from_storage.

    Args:
        session_file: Path to the session JSON file
        storage_dir: Path to OpenCode storage root

    Returns:
        ParsedSession if parsing succeeds, None if metadata is invalid.
    """
    metadata = parse_session_metadata_from_file(session_file)
    if metadata is None:
        return None

    messages = parse_session_messages_from_storage(metadata.session_id, storage_dir)

    return ParsedSession(
        session_id=metadata.session_id,
        repo_path=metadata.repo_path,
        timestamp=metadata.timestamp,
        parent_session_id=metadata.parent_session_id,
        messages=messages,
    )


def parse_session_metadata(session_path: Path) -> SessionMetadata | None:
    """Parse metadata from a session's session.json file (legacy format).

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


def parse_session_messages(session_path: Path) -> list[dict[str, str]]:
    """Parse messages from a session's messages.json file.

    Reads the conversation transcript from messages.json and returns
    the list of message objects. Each message should have 'role' and
    'content' keys.

    Args:
        session_path: Path to the session directory (not the JSON file)

    Returns:
        List of message dicts with 'role' and 'content' keys.
        Returns empty list if file is missing, invalid, or empty.

    Note:
        Defensively handles missing or malformed messages. Messages
        without content are skipped. Messages without role default to
        "unknown".
    """
    messages_json = session_path / "messages.json"
    session_id = session_path.name

    if not messages_json.exists():
        logger.warning(
            "messages_json_not_found",
            session_id=session_id,
            path=str(messages_json),
        )
        return []

    try:
        with open(messages_json, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(
            "messages_json_invalid",
            session_id=session_id,
            error=str(e),
        )
        return []
    except OSError as e:
        logger.warning(
            "messages_json_read_error",
            session_id=session_id,
            error=str(e),
        )
        return []

    if not isinstance(data, list):
        logger.warning(
            "messages_json_not_array",
            session_id=session_id,
            actual_type=type(data).__name__,
        )
        return []

    # Filter and normalize messages
    messages: list[dict[str, str]] = []
    for i, msg in enumerate(data):
        if not isinstance(msg, dict):
            logger.debug(
                "message_not_object",
                session_id=session_id,
                index=i,
                actual_type=type(msg).__name__,
            )
            continue

        content = msg.get("content", "")
        if not content:
            # Skip messages with empty content
            continue

        role = msg.get("role", "unknown")
        messages.append({"role": str(role), "content": str(content)})

    logger.debug(
        "messages_parsed",
        session_id=session_id,
        message_count=len(messages),
        original_count=len(data),
    )

    return messages


@dataclass
class ParsedSession:
    """A fully parsed session ready for chunking and embedding.

    Combines metadata from session.json with messages from messages.json.
    This is the intermediate representation before chunking.

    Attributes:
        session_id: OpenCode session ID
        repo_path: Absolute path to repository where session occurred
        timestamp: Unix timestamp of session creation
        parent_session_id: ID of parent session if this was a fork
        messages: List of message dicts with 'role' and 'content' keys
    """

    session_id: str
    repo_path: str
    timestamp: int
    parent_session_id: str | None
    messages: list[dict[str, str]]


def parse_session(session_path: Path) -> ParsedSession | None:
    """Parse a complete session including metadata and messages.

    Combines parse_session_metadata and parse_session_messages into
    a single ParsedSession object ready for chunking.

    Args:
        session_path: Path to the session directory

    Returns:
        ParsedSession if parsing succeeds, None if metadata is missing
        or invalid. Messages may be empty if messages.json is missing.
    """
    metadata = parse_session_metadata(session_path)
    if metadata is None:
        return None

    messages = parse_session_messages(session_path)

    return ParsedSession(
        session_id=metadata.session_id,
        repo_path=metadata.repo_path,
        timestamp=metadata.timestamp,
        parent_session_id=metadata.parent_session_id,
        messages=messages,
    )


def chunk_session(
    session: ParsedSession,
    model_used: str,
    config: ChunkingConfig | None = None,
) -> list[SessionChunk]:
    """Chunk a parsed session into SessionChunk objects for embedding.

    Takes a ParsedSession and splits its messages into chunks using
    the token-based chunker. Each chunk is converted to a SessionChunk
    with all required metadata for storage in LanceDB.

    Args:
        session: A fully parsed session with metadata and messages
        model_used: Identifier of the embedding model that will be used
        config: Optional chunking configuration

    Returns:
        List of SessionChunk objects ready for embedding. The embedding
        field is left as empty list - caller must populate embeddings.
        Returns empty list if session has no messages.

    Note:
        Chunks are created with empty embedding vectors. The caller is
        responsible for generating embeddings before storing in LanceDB.
    """
    if not session.messages:
        logger.debug(
            "session_has_no_messages",
            session_id=session.session_id,
        )
        return []

    # Chunk the messages using the token-based chunker
    chunks = chunk_messages(session.messages, config)

    if not chunks:
        logger.debug(
            "session_produced_no_chunks",
            session_id=session.session_id,
        )
        return []

    # Convert Chunk objects to SessionChunk objects
    session_chunks: list[SessionChunk] = []
    for chunk in chunks:
        chunk_id = f"{session.session_id}_chunk_{chunk.index}"
        session_chunks.append(
            SessionChunk(
                id=chunk_id,
                session_id=session.session_id,
                repo_path=session.repo_path,
                chunk_index=chunk.index,
                chunk_text=chunk.text,
                embedding=[],  # To be filled by embedding provider
                timestamp=session.timestamp,
                model_used=model_used,
                token_count=chunk.token_count,
            )
        )

    logger.info(
        "session_chunked",
        session_id=session.session_id,
        chunk_count=len(session_chunks),
        total_tokens=sum(c.token_count for c in session_chunks),
    )

    return session_chunks


def load_sync_state(sync_state_path: Path) -> SyncState:
    """Load sync state from JSON file.

    Args:
        sync_state_path: Path to the sync state JSON file.

    Returns:
        SyncState object. Returns empty state if file doesn't exist or is invalid.
    """
    sync_state_path = sync_state_path.expanduser()

    if not sync_state_path.exists():
        logger.info("sync_state_not_found", path=str(sync_state_path))
        return SyncState(last_sync=0, sessions={})

    try:
        with open(sync_state_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(
            "sync_state_invalid_json", path=str(sync_state_path), error=str(e)
        )
        return SyncState(last_sync=0, sessions={})
    except OSError as e:
        logger.warning("sync_state_read_error", path=str(sync_state_path), error=str(e))
        return SyncState(last_sync=0, sessions={})

    if not isinstance(data, dict):
        logger.warning(
            "sync_state_not_object",
            path=str(sync_state_path),
            actual_type=type(data).__name__,
        )
        return SyncState(last_sync=0, sessions={})

    last_sync = data.get("last_sync", 0)
    if not isinstance(last_sync, int):
        last_sync = int(last_sync) if isinstance(last_sync, (float, str)) else 0

    sessions = data.get("sessions", {})
    if not isinstance(sessions, dict):
        sessions = {}

    return SyncState(last_sync=last_sync, sessions=sessions)


def save_sync_state(sync_state: SyncState, sync_state_path: Path) -> None:
    """Save sync state to JSON file.

    Creates parent directories if they don't exist.

    Args:
        sync_state: The sync state to save.
        sync_state_path: Path to the sync state JSON file.

    Raises:
        OSError: If the file cannot be written.
    """
    sync_state_path = sync_state_path.expanduser()
    sync_state_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(sync_state)
    with open(sync_state_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(
        "sync_state_saved",
        path=str(sync_state_path),
        session_count=len(sync_state.sessions),
    )


@dataclass
class SyncResult:
    """Result of a sync operation.

    Attributes:
        sessions_processed: Number of sessions successfully processed.
        sessions_deleted: Number of sessions deleted from index.
        sessions_failed: Number of sessions that failed to process.
        chunks_added: Total number of chunks added to the database.
        errors: List of error messages for failed sessions.
    """

    sessions_processed: int = 0
    sessions_deleted: int = 0
    sessions_failed: int = 0
    chunks_added: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


@dataclass
class SyncProgress:
    """Progress update during sync.

    Attributes:
        current: Current session index (0-based).
        total: Total number of sessions to process.
        session_id: ID of the session being processed.
        status: Current status ("processing", "success", "failed", "deleted").
    """

    current: int
    total: int
    session_id: str
    status: str


@contextmanager
def acquire_lock(lock_path: Path) -> Generator[None, None, None]:
    """Acquire an exclusive file lock.

    Args:
        lock_path: Path to the lock file.

    Raises:
        OSError: If lock cannot be acquired (already locked).
    """
    lock_path = lock_path.expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as f:
        try:
            # Try to acquire exclusive, non-blocking lock
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        except OSError as e:
            logger.info("lock_acquisition_failed", path=str(lock_path), error=str(e))
            raise OSError(f"Could not acquire lock on {lock_path}") from e
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except OSError:
                pass


def sync_sessions(
    config: SmartForkConfig | None = None,
    *,
    force: bool = False,
    progress_callback: Callable[[SyncProgress], None] | None = None,
) -> SyncResult:
    """Run the full sync pipeline to index sessions.

    This is the main entry point for ingesting sessions into the vector database.
    It handles the complete pipeline:
    1. Acquire file lock to prevent concurrent syncs
    2. Load configuration and sync state
    3. Discover sessions from the sessions directory
    4. Determine which sessions are new/modified/deleted
    5. Delete removed sessions from the index
    6. For each session to process: parse → chunk → embed → store
    7. Save updated sync state

    Error handling:
    - Individual session failures don't stop the sync
    - Errors are collected and returned in SyncResult
    - Sync state is updated progressively so partial syncs can resume

    Rate limiting:
    - Embedding API calls respect config.embedding.rate_limit_ms between batches
    - This is handled internally by the embedding provider

    Args:
        config: Optional SmartForkConfig. If None, loads from default path.
        force: If True, re-index all sessions regardless of sync state.
        progress_callback: Optional callback for progress updates.

    Returns:
        SyncResult with counts of processed/deleted/failed sessions.

    Example:
        >>> from smart_fork.config import load_config
        >>> config = load_config()
        >>> result = sync_sessions(config)
        >>> print(f"Processed {result.sessions_processed} sessions")
    """
    # Load config if not provided
    if config is None:
        config = load_config()

    result = SyncResult()

    # Define lock path next to sync state
    lock_path = config.paths.sync_state_path.parent / "sync.lock"

    try:
        with acquire_lock(lock_path):
            return _sync_sessions_impl(config, force, progress_callback)
    except OSError:
        # Lock acquisition failed
        if result.errors is not None:
            result.errors.append("Sync already in progress (lock held)")
        return result


def _sync_sessions_impl(
    config: SmartForkConfig,
    force: bool,
    progress_callback: Callable[[SyncProgress], None] | None,
) -> SyncResult:
    """Internal implementation of sync_sessions, executed under lock."""
    result = SyncResult()

    # Load sync state
    sync_state = load_sync_state(config.paths.sync_state_path)

    # Discover sessions - use legacy or actual format based on config
    use_legacy_format = config.paths.is_legacy_mode()

    if use_legacy_format:
        # Legacy format: sessions_dir/ses_*/session.json + messages.json
        assert config.paths.sessions_dir is not None
        sessions = discover_sessions(config.paths.sessions_dir)
        storage_dir = None
        log_path = str(config.paths.sessions_dir)
    else:
        # Actual OpenCode format: storage/session/<project>/ses_*.json
        sessions = discover_sessions_from_storage(config.paths.storage_dir)
        storage_dir = config.paths.storage_dir
        log_path = str(config.paths.storage_dir)

    # Determine what needs to be synced
    # Note: Even if no sessions found, we still need to check for deleted sessions
    sessions_to_process, session_ids_to_delete = get_sessions_to_sync(
        sessions, sync_state.sessions, force
    )

    if not sessions and not session_ids_to_delete:
        logger.info("no_sessions_found", sessions_dir=log_path)
        return result

    if not sessions_to_process and not session_ids_to_delete:
        logger.info("nothing_to_sync")
        return result

    # Open database
    try:
        db = ChunkDatabase.open(config.paths.lance_path)
    except DatabaseError as e:
        logger.error("database_open_failed", error=str(e))
        result.errors = [f"Failed to open database: {e}"]
        return result

    # Create embedding provider
    try:
        provider = create_provider(config.embedding)
        model_name = provider.model_name()
        logger.info("embedding_provider_created", model=model_name)
    except EmbeddingError as e:
        logger.error("embedding_provider_failed", error=str(e))
        result.errors = [f"Failed to create embedding provider: {e}"]
        db.close()
        return result

    # Handle force mode: drop existing table to start fresh
    if force:
        try:
            dropped = db.drop_table()
            if dropped:
                logger.info("table_dropped_for_force_sync")
        except DatabaseError as e:
            logger.warning("table_drop_failed", error=str(e))

    # Delete removed sessions
    for session_id in session_ids_to_delete:
        if progress_callback:
            progress_callback(
                SyncProgress(
                    current=result.sessions_deleted,
                    total=len(session_ids_to_delete),
                    session_id=session_id,
                    status="deleted",
                )
            )
        try:
            deleted_count = db.delete_by_session(session_id)
            logger.debug(
                "session_deleted",
                session_id=session_id,
                chunks_deleted=deleted_count,
            )
            # Remove from sync state
            sync_state.sessions.pop(session_id, None)
            result.sessions_deleted += 1
        except DatabaseError as e:
            logger.warning(
                "session_delete_failed",
                session_id=session_id,
                error=str(e),
            )

    # Process sessions
    total_to_process = len(sessions_to_process)
    for i, session_info in enumerate(sessions_to_process):
        session_id = session_info.session_id

        if progress_callback:
            progress_callback(
                SyncProgress(
                    current=i,
                    total=total_to_process,
                    session_id=session_id,
                    status="processing",
                )
            )

        try:
            # Parse session - use appropriate parser based on format
            if use_legacy_format:
                # Legacy: path is a directory with session.json + messages.json
                parsed = parse_session(session_info.path)
            else:
                # Actual: path is a session JSON file, messages in storage
                assert storage_dir is not None
                parsed = parse_session_from_storage(session_info.path, storage_dir)

            if parsed is None:
                logger.warning(
                    "session_parse_failed",
                    session_id=session_id,
                )
                result.sessions_failed += 1
                if result.errors is not None:
                    result.errors.append(f"{session_id}: Failed to parse session")
                continue

            # Chunk session
            chunks = chunk_session(parsed, model_name, config.chunking)
            if not chunks:
                # Session has no content (empty messages)
                logger.debug(
                    "session_has_no_chunks",
                    session_id=session_id,
                )
                # Still mark as synced to avoid re-processing
                sync_state.sessions[session_id] = session_info.last_modified
                continue

            # Generate embeddings for all chunks
            try:
                chunk_texts = [c.chunk_text for c in chunks]
                embeddings = provider.embed(chunk_texts)

                # Fill in embeddings
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    # Create new chunk with embedding (SessionChunk is immutable-ish)
                    chunk.embedding = embedding

            except EmbeddingError as e:
                logger.warning(
                    "session_embedding_failed",
                    session_id=session_id,
                    error=str(e),
                )
                result.sessions_failed += 1
                if result.errors is not None:
                    result.errors.append(f"{session_id}: Embedding failed: {e}")
                continue

            # Delete old chunks for this session (in case of re-index)
            try:
                db.delete_by_session(session_id)
            except DatabaseError:
                pass  # Ignore - table may not exist yet

            # Add new chunks to database
            try:
                added = db.add_chunks(chunks)
                result.chunks_added += added
            except DatabaseError as e:
                logger.warning(
                    "session_store_failed",
                    session_id=session_id,
                    error=str(e),
                )
                result.sessions_failed += 1
                if result.errors is not None:
                    result.errors.append(f"{session_id}: Storage failed: {e}")
                continue

            # Update sync state
            sync_state.sessions[session_id] = session_info.last_modified
            result.sessions_processed += 1

            if progress_callback:
                progress_callback(
                    SyncProgress(
                        current=i + 1,
                        total=total_to_process,
                        session_id=session_id,
                        status="success",
                    )
                )

            logger.debug(
                "session_indexed",
                session_id=session_id,
                chunks=len(chunks),
            )

        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                "session_unexpected_error",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            result.sessions_failed += 1
            if result.errors is not None:
                result.errors.append(f"{session_id}: Unexpected error: {e}")

            if progress_callback:
                progress_callback(
                    SyncProgress(
                        current=i + 1,
                        total=total_to_process,
                        session_id=session_id,
                        status="failed",
                    )
                )

    # Save sync state
    sync_state.last_sync = int(time.time())
    try:
        save_sync_state(sync_state, config.paths.sync_state_path)
    except OSError as e:
        logger.error("sync_state_save_failed", error=str(e))
        if result.errors is not None:
            result.errors.append(f"Failed to save sync state: {e}")

    # Close database
    db.close()

    logger.info(
        "sync_complete",
        sessions_processed=result.sessions_processed,
        sessions_deleted=result.sessions_deleted,
        sessions_failed=result.sessions_failed,
        chunks_added=result.chunks_added,
    )

    return result

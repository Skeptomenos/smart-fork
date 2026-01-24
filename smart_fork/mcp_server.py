"""MCP server for Smart Fork.

Provides semantic session discovery as MCP tools that integrate
natively with OpenCode and other MCP clients.

Tools:
- smart_fork_search: Search for sessions by natural language query
- smart_fork_status: Get index statistics
- smart_fork_sync: Trigger a sync of new sessions

Usage:
    Run as MCP server:
        uv run python -m smart_fork.mcp_server

    Or via the CLI entry point:
        uv run smart-fork-mcp
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from smart_fork.config import load_config
from smart_fork.db import ChunkDatabase
from smart_fork.ingest import SyncProgress, sync_sessions
from smart_fork.query import QueryResult, search_sessions

# Initialize MCP server
mcp = FastMCP(
    "smart-fork",
    instructions="Semantic session discovery for OpenCode. Search past sessions to find relevant context for forking.",
)


def _format_time_ago(timestamp: int) -> str:
    """Format a Unix timestamp as a human-readable 'time ago' string."""
    now = int(time.time())
    diff = now - timestamp

    if diff < 60:
        return "just now"
    elif diff < 3600:
        mins = diff // 60
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    elif diff < 86400:
        hours = diff // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff < 604800:
        days = diff // 86400
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = diff // 604800
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"


def _format_search_results(result: QueryResult) -> dict[str, Any]:
    """Format search results for MCP response."""
    matches = []
    for i, match in enumerate(result.matches, 1):
        matches.append(
            {
                "rank": i,
                "session_id": match.session_id,
                "score": round(match.score * 100, 1),
                "score_percent": f"{match.score * 100:.0f}%",
                "repo": match.repo_name,
                "repo_path": match.repo_path,
                "time_ago": _format_time_ago(match.timestamp),
                "timestamp": match.timestamp,
                "snippet": match.best_snippet,
                "chunk_count": match.chunk_count,
                "fork_command": f"opencode --session {match.session_id}",
            }
        )

    return {
        "matches": matches,
        "query_time_ms": round(result.query_time_ms, 0),
        "total_chunks_searched": result.total_chunks_searched,
        "top_result_fork_command": (
            f"opencode --session {result.matches[0].session_id}"
            if result.matches
            else None
        ),
    }


@mcp.tool()
def smart_fork_search(
    query: str,
    scope: str = "global",
    repo_path: str | None = None,
) -> dict[str, Any]:
    """Search for OpenCode sessions semantically similar to a query.

    Use this tool to find past sessions that contain relevant context for
    the current task. Returns ranked results with fork commands.

    Args:
        query: Natural language description of what you're looking for.
               Example: "implement webhook signature verification"
        scope: Search scope - "global" for all sessions, "repo" for current repo only.
        repo_path: Optional path to filter results to a specific repository.
                   Only used when scope is "repo".

    Returns:
        Dictionary with:
        - matches: List of matching sessions with scores, snippets, and fork commands
        - query_time_ms: Time taken for the search
        - top_result_fork_command: Fork command for the best match

    Example:
        >>> smart_fork_search("API rate limiting implementation")
        {
            "matches": [
                {
                    "rank": 1,
                    "session_id": "ses_abc123",
                    "score_percent": "73%",
                    "repo": "my-api",
                    "time_ago": "2 days ago",
                    "snippet": "Implemented rate limiting using token bucket...",
                    "fork_command": "opencode --session ses_abc123"
                },
                ...
            ],
            "top_result_fork_command": "opencode --session ses_abc123"
        }
    """
    # Determine repo filter
    filter_repo: str | None = None
    if scope == "repo" and repo_path:
        filter_repo = repo_path

    try:
        result = search_sessions(query, repo_path=filter_repo)
        return _format_search_results(result)
    except Exception as e:
        return {
            "error": str(e),
            "matches": [],
            "query_time_ms": 0,
            "top_result_fork_command": None,
        }


@mcp.tool()
def smart_fork_status() -> dict[str, Any]:
    """Get the current status of the Smart Fork index.

    Returns statistics about the indexed sessions and database.

    Returns:
        Dictionary with:
        - sessions: Number of indexed sessions
        - chunks: Total number of chunks in the database
        - last_sync: ISO timestamp of last sync
        - last_sync_ago: Human-readable time since last sync
        - database_path: Path to the LanceDB database
    """
    try:
        config = load_config()
        db_path = config.paths.lance_path

        # Get database stats
        if db_path.exists():
            db = ChunkDatabase.open(db_path)
            session_count = db.count_sessions()
            chunk_count = db.count_chunks()
        else:
            session_count = 0
            chunk_count = 0

        # Get last sync time from sync state
        sync_state_path = config.paths.sync_state_path
        last_sync: str | None = None
        last_sync_ago: str | None = None

        if sync_state_path.exists():
            import json

            with open(sync_state_path) as f:
                state = json.load(f)
            last_sync_ts = state.get("last_sync_time")
            if last_sync_ts:
                last_sync = datetime.fromtimestamp(
                    last_sync_ts, tz=timezone.utc
                ).isoformat()
                last_sync_ago = _format_time_ago(last_sync_ts)

        return {
            "sessions": session_count,
            "chunks": chunk_count,
            "last_sync": last_sync,
            "last_sync_ago": last_sync_ago,
            "database_path": str(db_path),
        }
    except Exception as e:
        return {
            "error": str(e),
            "sessions": 0,
            "chunks": 0,
            "last_sync": None,
            "database_path": None,
        }


@mcp.tool()
def smart_fork_sync(
    full: bool = False,
) -> dict[str, Any]:
    """Sync OpenCode sessions to the Smart Fork index.

    Discovers new and modified sessions and indexes them for search.
    This operation can take several minutes for large session histories.

    Args:
        full: If True, re-index all sessions. If False (default), only sync
              new and modified sessions since last sync.

    Returns:
        Dictionary with:
        - sessions_synced: Number of sessions processed
        - chunks_added: Number of new chunks added
        - sessions_failed: Number of sessions that failed to process
        - duration_seconds: Time taken for the sync
        - errors: List of error messages (if any)
    """
    start_time = time.time()
    errors: list[str] = []

    # Progress callback to collect errors
    def on_progress(progress: SyncProgress) -> None:
        if progress.status == "failed":
            errors.append(f"{progress.session_id}: failed")

    try:
        result = sync_sessions(
            force=full,
            progress_callback=on_progress,
        )

        duration = time.time() - start_time

        return {
            "sessions_synced": result.sessions_processed,
            "chunks_added": result.chunks_added,
            "sessions_failed": result.sessions_failed,
            "sessions_deleted": result.sessions_deleted,
            "duration_seconds": round(duration, 1),
            "errors": errors[:10] if errors else [],  # Limit to first 10 errors
            "has_more_errors": len(errors) > 10,
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            "error": str(e),
            "sessions_synced": 0,
            "chunks_added": 0,
            "sessions_failed": 0,
            "duration_seconds": round(duration, 1),
            "errors": errors[:10] if errors else [],
        }


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()

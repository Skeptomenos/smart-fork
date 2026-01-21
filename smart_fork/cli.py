"""CLI entry point for Smart Fork.

This module provides the `smart-fork` command with subcommands:
- sync: Index session transcripts into LanceDB
- search: Query sessions by natural language
- status: Show index statistics
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

if TYPE_CHECKING:
    from smart_fork.ingest import SyncProgress as SyncProgressType
    from smart_fork.query import QueryResult

# Lazy console initialization to avoid overhead when not needed
_console: Console | None = None


def _get_console() -> Console:
    """Get or create the Rich console for output."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def _format_time_ago(timestamp: int) -> str:
    """Format a Unix timestamp as a human-readable relative time.

    Examples:
        "just now", "5 minutes", "2 hours", "3 days", "2 weeks", "1 month"
    """
    import time

    now = int(time.time())
    diff_seconds = max(0, now - timestamp)

    # Time units in seconds
    minute = 60
    hour = minute * 60
    day = hour * 24
    week = day * 7
    month = day * 30  # Approximate

    if diff_seconds < minute:
        return "just now"
    elif diff_seconds < hour:
        minutes = diff_seconds // minute
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    elif diff_seconds < day:
        hours = diff_seconds // hour
        return f"{hours} hour{'s' if hours != 1 else ''}"
    elif diff_seconds < week:
        days = diff_seconds // day
        return f"{days} day{'s' if days != 1 else ''}"
    elif diff_seconds < month:
        weeks = diff_seconds // week
        return f"{weeks} week{'s' if weeks != 1 else ''}"
    else:
        months = diff_seconds // month
        return f"{months} month{'s' if months != 1 else ''}"


def _display_search_results(
    result: QueryResult,
    repo_filter: str | None,
    console: Console,
) -> None:
    """Display search results as a Rich table.

    Formats the QueryResult matches into a table showing rank, score,
    repo name, relative time, and context snippet. Also outputs the
    fork command for the top result.

    Args:
        result: QueryResult from search_sessions()
        repo_filter: Repo path filter used (for display purposes)
        console: Rich Console for output
    """
    from rich.table import Table

    if not result.matches:
        console.print("[dim]No matching sessions found.[/dim]")
        console.print()
        console.print("[dim]Tips:[/dim]")
        console.print("  • Try broader search terms")
        console.print("  • Run [cyan]smart-fork sync[/cyan] to index new sessions")
        if repo_filter:
            console.print(
                "  • Remove [cyan]--scope repo[/cyan] to search all repositories"
            )
        return

    # Build table with columns: #, Score, Repo, When, Context
    # When filtering by repo, omit the Repo column since all results are same repo
    show_repo_column = repo_filter is None

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Score", justify="right", width=6)
    if show_repo_column:
        table.add_column("Repo", style="cyan", no_wrap=True)
    table.add_column("When", style="yellow", no_wrap=True)
    table.add_column("Context", overflow="ellipsis")

    for i, match in enumerate(result.matches, 1):
        # Format score as percentage (e.g., "94%")
        score_str = f"{match.score * 100:.0f}%"

        # Format relative time
        when_str = _format_time_ago(match.timestamp)

        # Truncate snippet for display (table will handle overflow)
        context = match.best_snippet.replace("\n", " ").strip()
        if len(context) > 60:
            context = context[:57] + "..."

        if show_repo_column:
            table.add_row(str(i), score_str, match.repo_name, when_str, context)
        else:
            table.add_row(str(i), score_str, when_str, context)

    console.print(table)
    console.print()

    # Output fork command for top result
    top_match = result.matches[0]
    console.print(
        f"[bold green]→[/bold green] opencode --session {top_match.session_id}"
    )
    console.print()
    console.print(f"[dim]Query completed in {result.query_time_ms:.0f}ms[/dim]")


@click.group()
@click.version_option()
def main() -> None:
    """Semantic session discovery for OpenCode.

    Find the most relevant sessions to fork from using natural language search.
    """
    pass


@main.command()
@click.option("--force", is_flag=True, help="Re-index all sessions")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def sync(force: bool, quiet: bool) -> None:
    """Index session transcripts into LanceDB.

    Discovers sessions in ~/.local/share/opencode/sessions/ and indexes
    them for semantic search. By default, only new or modified sessions
    are processed (incremental sync).

    Examples:
        smart-fork sync           # Incremental sync
        smart-fork sync --force   # Re-index everything
        smart-fork sync --quiet   # Silent mode for cron/scripts
    """
    # Lazy imports to speed up CLI startup
    from smart_fork.config import load_config
    from smart_fork.ingest import SyncProgress, SyncResult, sync_sessions

    console = _get_console()

    try:
        config = load_config()
    except ValueError as e:
        if not quiet:
            console.print(f"[red]Error loading config:[/red] {e}")
        sys.exit(1)

    # Track progress state for the callback
    progress_bar: Progress | None = None
    task_id: TaskID | None = None

    def progress_callback(progress: SyncProgressType) -> None:
        """Update progress bar during sync."""
        nonlocal progress_bar, task_id

        if quiet:
            return

        # Initialize progress bar on first call if we have sessions to process
        if progress_bar is None and progress.total > 0:
            progress_bar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )
            progress_bar.start()
            task_id = progress_bar.add_task(
                "[cyan]Syncing sessions...", total=progress.total
            )

        # Update progress bar
        if progress_bar is not None and task_id is not None:
            # Show current session being processed
            status_emoji = {
                "processing": "⏳",
                "success": "✓",
                "failed": "✗",
                "deleted": "🗑",
            }.get(progress.status, "")

            progress_bar.update(
                task_id,
                description=f"[cyan]{status_emoji} {progress.session_id[:16]}...",
                completed=progress.current + 1,
            )

    if not quiet:
        action = "Re-indexing" if force else "Syncing"
        console.print(f"[bold]{action} sessions...[/bold]")

    try:
        result: SyncResult = sync_sessions(
            config=config,
            force=force,
            progress_callback=progress_callback,
        )
    except Exception as e:
        if progress_bar is not None:
            progress_bar.stop()
        if not quiet:
            console.print(f"[red]Sync failed:[/red] {e}")
        sys.exit(1)
    finally:
        if progress_bar is not None:
            progress_bar.stop()

    # Report results
    if not quiet:
        console.print()  # Blank line after progress bar

        # Check if anything happened (processed, deleted, or failed)
        nothing_happened = (
            result.sessions_processed == 0
            and result.sessions_deleted == 0
            and result.sessions_failed == 0
        )

        if nothing_happened:
            console.print("[dim]No new or modified sessions to sync.[/dim]")
        else:
            if result.sessions_processed > 0:
                console.print(
                    f"[green]✓[/green] Processed {result.sessions_processed} session(s), "
                    f"{result.chunks_added} chunks indexed"
                )
            if result.sessions_deleted > 0:
                console.print(
                    f"[yellow]🗑[/yellow] Removed {result.sessions_deleted} deleted session(s)"
                )
            if result.sessions_failed > 0:
                console.print(
                    f"[red]✗[/red] Failed to process {result.sessions_failed} session(s)"
                )
                # Show first few errors as hints
                if result.errors:
                    for error in result.errors[:3]:
                        console.print(f"  [dim]• {error}[/dim]")
                    if len(result.errors) > 3:
                        console.print(
                            f"  [dim]... and {len(result.errors) - 3} more errors[/dim]"
                        )

    # Exit with error code if any failures occurred
    if result.sessions_failed > 0:
        sys.exit(1)


@main.command()
@click.argument("query")
@click.option(
    "--scope",
    type=click.Choice(["global", "repo"]),
    default="global",
    help="Search scope: 'global' (all sessions) or 'repo' (current repo only)",
)
@click.option(
    "--repo",
    "repo_path",
    help="Filter results to a specific repository path",
)
def search(query: str, scope: str, repo_path: str | None) -> None:
    """Search sessions by natural language query.

    Searches the indexed sessions for content semantically similar to QUERY
    and returns the top matching sessions with relevance scores.

    Examples:
        smart-fork search "implement rate limiting"
        smart-fork search --scope repo "continue refactoring"
        smart-fork search --repo /path/to/repo "webhook handling"
    """
    # Lazy imports to speed up CLI startup
    import os
    from pathlib import Path

    from smart_fork.config import load_config
    from smart_fork.query import EmptyQueryError, QueryError, search_sessions

    console = _get_console()

    try:
        config = load_config()
    except ValueError as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        sys.exit(1)

    # Determine repo filter based on scope and --repo flag
    # --repo flag takes precedence over --scope repo
    effective_repo_path: str | None = None
    if repo_path:
        # Explicit --repo flag provided
        effective_repo_path = str(Path(repo_path).resolve())
    elif scope == "repo":
        # --scope repo uses current working directory
        effective_repo_path = os.getcwd()

    try:
        result = search_sessions(
            query,
            config=config,
            repo_path=effective_repo_path,
        )
    except EmptyQueryError:
        console.print("[red]Error:[/red] Query cannot be empty")
        sys.exit(1)
    except QueryError as e:
        console.print(f"[red]Search failed:[/red] {e}")
        sys.exit(1)

    # Display results
    _display_search_results(result, effective_repo_path, console)


@main.command()
def status() -> None:
    """Show index statistics.

    Displays information about the Smart Fork index including:
    - Number of indexed sessions
    - Total chunks in the database
    - Last sync timestamp
    - Database location

    Examples:
        smart-fork status
    """
    # Lazy imports to speed up CLI startup
    from datetime import datetime

    from smart_fork.config import load_config
    from smart_fork.db import ChunkDatabase
    from smart_fork.ingest import load_sync_state

    console = _get_console()

    try:
        config = load_config()
    except ValueError as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        sys.exit(1)

    # Open database and get stats
    db = ChunkDatabase.open(config.paths.lance_path)

    session_count = db.count_sessions()
    chunk_count = db.count_chunks()

    # Load sync state for last sync time
    sync_state = load_sync_state(config.paths.sync_state_path)

    # Display status
    console.print("[bold]Smart Fork Status[/bold]")
    console.print()

    if session_count == 0:
        console.print("[dim]No sessions indexed yet.[/dim]")
        console.print()
        console.print("Run [cyan]smart-fork sync[/cyan] to index your sessions.")
    else:
        console.print(f"[green]Sessions:[/green]  {session_count:,}")
        console.print(f"[green]Chunks:[/green]    {chunk_count:,}")

        if sync_state.last_sync > 0:
            # Format last sync time as human-readable
            last_sync_dt = datetime.fromtimestamp(sync_state.last_sync)
            last_sync_str = last_sync_dt.strftime("%Y-%m-%d %H:%M:%S")
            last_sync_ago = _format_time_ago(sync_state.last_sync)
            console.print(
                f"[green]Last sync:[/green] {last_sync_str} ({last_sync_ago} ago)"
            )
        else:
            console.print("[green]Last sync:[/green] [dim]Never[/dim]")

    console.print()
    console.print(f"[dim]Database:[/dim] {config.paths.lance_path}")


if __name__ == "__main__":
    main()

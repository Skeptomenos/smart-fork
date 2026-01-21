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

# Lazy console initialization to avoid overhead when not needed
_console: Console | None = None


def _get_console() -> Console:
    """Get or create the Rich console for output."""
    global _console
    if _console is None:
        _console = Console()
    return _console


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
@click.option("--scope", type=click.Choice(["global", "repo"]), default="global")
@click.option("--repo", help="Repo path to filter by")
def search(query: str, scope: str, repo: str | None) -> None:
    """Search sessions by natural language query."""
    # Implementation in Task 6.2
    click.echo(f"search command not yet implemented (Phase 6)")


@main.command()
def status() -> None:
    """Show index statistics."""
    # Implementation in Task 6.4
    click.echo("status command not yet implemented (Phase 6)")


if __name__ == "__main__":
    main()

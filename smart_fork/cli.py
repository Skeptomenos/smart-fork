"""CLI entry point for Smart Fork.

This module provides the `smart-fork` command with subcommands:
- sync: Index session transcripts into LanceDB
- search: Query sessions by natural language
- status: Show index statistics

Full implementation in Phase 6.
"""

import click


@click.group()
@click.version_option()
def main() -> None:
    """Semantic session discovery for OpenCode.

    Find the most relevant sessions to fork from using natural language search.
    """
    pass


@main.command()
@click.option("--force", is_flag=True, help="Re-index all sessions")
@click.option("--quiet", is_flag=True, help="Suppress output")
def sync(force: bool, quiet: bool) -> None:
    """Index session transcripts into LanceDB."""
    # Implementation in Task 6.1
    click.echo("sync command not yet implemented (Phase 6)")


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

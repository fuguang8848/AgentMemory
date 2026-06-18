"""Typer CLI main entry point for AgentMemory 2.0.

Per ARCHITECTURE.md §8 (lines 783-794):
    - app.py: typer app
    - Sub-commands: add, search, stats, decay, serve, migrate, doctor

Commands are dispatched to individual modules under commands/.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="agentmemory",
    help="AgentMemory 2.0 - Long-term memory for AI agents",
    add_completion=False,
)
console = Console()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output"),
    memory_dir: Optional[str] = typer.Option(
        None, "--memory-dir", help="Override default memory directory"
    ),
):
    """AgentMemory 2.0 CLI."""
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")
    pass


# Import and register sub-commands
from agentmemory.cli.commands.add import app as add_app
from agentmemory.cli.commands.search import app as search_app
from agentmemory.cli.commands.stats import app as stats_app
from agentmemory.cli.commands.doctor import app as doctor_app
from agentmemory.cli.commands.serve import app as serve_app

app.add_typer(add_app, name="add", help="Add memory entries")
app.add_typer(search_app, name="search", help="Search memory entries")
app.add_typer(stats_app, name="stats", help="Show memory statistics")
app.add_typer(doctor_app, name="doctor", help="Self-diagnosis and health checks")
app.add_typer(serve_app, name="serve", help="Start HTTP/MCP server")


def run():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()

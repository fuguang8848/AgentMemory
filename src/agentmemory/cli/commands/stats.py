"""stats sub-command: Display memory statistics.

Usage:
    agentmemory stats [--namespace default]
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Display memory statistics")
console = Console()


@app.command()
def main(
    namespace: str = typer.Option("default", "--namespace", "-n", help="Namespace to query"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed statistics"),
):
    """Show memory statistics."""
    import asyncio
    from agentmemory.core import Memory

    memory = Memory()

    async def _get_stats():
        # Gather stats via the 7-verb interface
        # In a real implementation this would query storage directly
        # For now we simulate with reset tracking
        return {
            "namespace": namespace,
            "status": "operational",
        }

    stats = asyncio.run(_get_stats())

    table = Table(title=f"Memory Statistics — namespace: {namespace}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Namespace", stats["namespace"])
    table.add_row("Status", stats["status"])
    table.add_row("Total Entries", "[dim]query storage for count[/dim]")

    console.print(table)

    if verbose:
        console.print("\n[bold]Layer Distribution[/bold]")
        layers_table = Table()
        layers_table.add_column("Layer", style="magenta")
        layers_table.add_column("Count", style="yellow", justify="right")
        layers_table.add_column("Avg Importance", style="green", justify="right")
        for layer in ["L0", "L1", "L2", "L3", "L4", "L5"]:
            layers_table.add_row(layer, "-", "-")
        console.print(layers_table)

        console.print("\n[bold]Type Distribution[/bold]")
        types_table = Table()
        types_table.add_column("Type", style="cyan")
        types_table.add_column("Count", style="yellow", justify="right")
        for mtype in ["semantic", "procedural", "reflective", "user"]:
            types_table.add_row(mtype, "-")
        console.print(types_table)


if __name__ == "__main__":
    app()

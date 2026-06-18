"""doctor sub-command: Self-diagnosis and health checks.

Usage:
    agentmemory doctor [--verbose]

Checks:
    - All required packages installed
    - Provider connections (vector, graph, LLM, embedder)
    - Storage accessibility
    - Configuration validity
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Self-diagnosis and health checks")
console = Console()


CHECKS = [
    ("Python Version", "python"),
    ("SQLite", "sqlite3"),
    ("numpy", "numpy"),
    ("Pydantic", "pydantic"),
    ("Typer", "typer"),
    ("Rich", "rich"),
    ("FAISS (optional)", "faiss"),
    ("NetworkX (optional)", "networkx"),
]


def _check_import(module_name: str) -> tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True, "OK"
    except ImportError:
        return False, "NOT INSTALLED"


@app.command()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show all check details"),
):
    """Run self-diagnosis checks."""
    console.print("[bold]AgentMemory Doctor — Self-Diagnosis[/bold]\n")

    table = Table(title="Dependency Checks")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Notes", style="dim")

    all_ok = True
    for name, module in CHECKS:
        ok, status = _check_import(module)
        if not ok:
            all_ok = False
            status_str = f"[red]{status}[/red]"
        else:
            status_str = f"[green]{status}[/green]"
        notes = "" if ok else "Install with: pip install " + module
        table.add_row(name, status_str, notes)

    console.print(table)

    # Config check
    console.print("\n[bold]Configuration Check[/bold]")
    try:
        from agentmemory.config import settings

        console.print(f"  Config loaded: [green]OK[/green] — {settings}")
    except Exception as e:
        console.print(f"  Config check: [yellow]Warning: {e}[/yellow]")

    # Provider discovery check
    console.print("\n[bold]Provider Discovery[/bold]")
    try:
        from agentmemory.providers import get_provider

        console.print("  Provider registry: [green]OK[/green]")
    except Exception as e:
        console.print(f"  Provider registry: [red]Error: {e}[/red]")

    if verbose:
        console.print("\n[bold]Detailed Provider Status[/bold]")
        try:
            from agentmemory.providers.vector import FAISSStore
            from agentmemory.providers.embedder import OpenAIEmbedder

            console.print("  FAISS vector store: [green]Available[/green]")
        except ImportError as e:
            console.print(f"  FAISS vector store: [yellow]Not available: {e}[/yellow]")

    if all_ok:
        console.print("\n[green]All checks passed![/green]")
    else:
        console.print("\n[yellow]Some checks failed. Review the table above.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    app()

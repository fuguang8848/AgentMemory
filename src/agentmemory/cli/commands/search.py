"""search sub-command: Search memory entries.

Usage:
    agentmemory search "query text" [--top-k 5] [--type semantic]
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Search memory entries")
console = Console()


@app.command()
def main(
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, max=100, help="Number of results"),
    memory_type: str = typer.Option(
        None, "--type", "-t", help="Filter by memory type"
    ),
    layer: str = typer.Option(None, "--layer", "-l", help="Filter by memory layer"),
    min_score: float = typer.Option(
        0.0, "--min-score", "-s", min=0.0, max=1.0, help="Minimum relevance score"
    ),
    rerank: bool = typer.Option(False, "--rerank", help="Enable reranking"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Namespace to search"),
):
    """Search memory entries matching the query."""
    from agentmemory.core import Memory, SearchQuery, MemoryType, MemoryLayer

    memory = Memory()

    filter_type = None
    if memory_type:
        try:
            filter_type = [MemoryType(memory_type.lower())]
        except ValueError:
            console.print(f"[red]Invalid memory type: {memory_type}[/red]")
            raise typer.Exit(1)

    filter_layer = None
    if layer:
        try:
            filter_layer = [MemoryLayer(layer.upper())]
        except ValueError:
            console.print(f"[red]Invalid layer: {layer}[/red]")
            raise typer.Exit(1)

    search_query = SearchQuery(
        text=query,
        top_k=top_k,
        filter_type=filter_type,
        filter_layer=filter_layer,
        min_score=min_score,
        rerank=rerank,
        namespace=namespace,
    )

    import asyncio

    async def _search():
        return await memory.search(search_query)

    results = asyncio.run(_search())

    if not results:
        console.print("[dim]No results found[/dim]")
        return

    table = Table(title=f"Search results for: {query!r}")
    table.add_column("ID", style="cyan", no_wrap=False)
    table.add_column("Score", style="yellow", justify="right")
    table.add_column("Layer", style="magenta")
    table.add_column("Content", style="white")
    table.add_column("Tags", style="dim")

    for result in results:
        item = result.item
        tags_str = ", ".join(item.tags) if item.tags else "-"
        content_preview = item.content[:80] + ("..." if len(item.content) > 80 else "")
        table.add_row(
            item.id[:12] + "...",
            f"{result.score:.3f}",
            item.layer.value,
            content_preview,
            tags_str,
        )

    console.print(table)
    console.print(f"[dim]{len(results)} result(s)[/dim]")


if __name__ == "__main__":
    app()

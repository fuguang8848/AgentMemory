"""add sub-command: Add memory entries.

Usage:
    agentmemory add "memory content" [--type semantic] [--layer L3]
    agentmemory add --file /path/to/memories.txt [--batch]
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Add memory entries to the memory store")
console = Console()


@app.command()
def main(
    content: Optional[str] = typer.Argument(None, help="Memory content to add"),
    file_path: Optional[str] = typer.Option(
        None, "--file", "-f", help="Path to file containing memory content (one per line)"
    ),
    memory_type: str = typer.Option(
        "semantic", "--type", "-t", help="Memory type: semantic, procedural, reflective, user"
    ),
    layer: str = typer.Option("L3", "--layer", "-l", help="Memory layer: L0-L5"),
    importance: float = typer.Option(
        0.5, "--importance", "-i", min=0.0, max=1.0, help="Importance score 0.0-1.0"
    ),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
    batch: bool = typer.Option(False, "--batch", help="Batch mode: content is newline-separated"),
):
    """Add one or more memory entries."""
    from agentmemory.core import Memory, MemoryType, MemoryLayer

    memory = Memory()

    if file_path:
        import pathlib

        path = pathlib.Path(file_path)
        if not path.exists():
            console.print(f"[red]File not found: {file_path}[/red]")
            raise typer.Exit(1)
        contents = path.read_text(encoding="utf-8").splitlines()
        entries = [c.strip() for c in contents if c.strip()]
        console.print(f"[dim]Adding {len(entries)} entries from {file_path}...[/dim]")
    elif batch and content:
        entries = [c.strip() for c in content.split("\\n") if c.strip()]
    elif content:
        entries = [content]
    else:
        console.print("[red]Error: provide content argument or --file option[/red]")
        raise typer.Exit(1)

    try:
        type_enum = MemoryType(memory_type.lower())
    except ValueError:
        console.print(f"[red]Invalid memory type: {memory_type}[/red]")
        raise typer.Exit(1)

    try:
        layer_enum = MemoryLayer(layer.upper())
    except ValueError:
        console.print(f"[red]Invalid layer: {layer}[/red]")
        raise typer.Exit(1)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    import asyncio

    async def _add_entries():
        results = []
        for entry in entries:
            result = await memory.add(
                content=entry,
                type=type_enum,
                layer=layer_enum,
                importance=importance,
                tags=tag_list,
            )
            results.extend(result)
        return results

    ids = asyncio.run(_add_entries())

    console.print(f"[green]Added {len(ids)} memory entries[/green]")
    for mid in ids:
        console.print(f"  [dim]ID: {mid}[/dim]")


if __name__ == "__main__":
    app()

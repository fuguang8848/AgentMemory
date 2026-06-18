"""serve sub-command: Start HTTP/MCP server.

Usage:
    agentmemory serve [--host 0.0.0.0] [--port 8000]
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Start HTTP/MCP server")
console = Console()


@app.command()
def main(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)"),
):
    """Start the AgentMemory HTTP server."""
    console.print(f"[bold]Starting AgentMemory server on {host}:{port}[/bold]")

    try:
        import uvicorn
    except ImportError:
        console.print("[red]Error: uvicorn not installed. Install with: pip install uvicorn[/red]")
        raise typer.Exit(1)

    # Import the ASGI app
    try:
        from agentmemory.server.launcher import get_app
    except ImportError:
        console.print("[yellow]Warning: agentmemory.server not found. Starting fallback HTTP server.[/yellow]")

        def fallback_app(scope, receive, send):
            import asyncio

            async def asgi_app(scope, receive, send):
                from pathlib import Path

                body = b""
                while True:
                    message = await receive()
                    if message["type"] == "http.request":
                        body += message.get("body", b"")
                    elif message["type"] == "http.disconnect":
                        break
                response_body = b"AgentMemory 2.0 Server Running"
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [[b"content-type", b"text/plain"]],
                })
                await send({
                    "type": "http.response.body",
                    "body": response_body,
                })
            asyncio.run(asgi_app(scope, receive, send))

        app_factory = fallback_app
    else:
        app_factory = get_app

    console.print(f"[dim]Workers: {workers} | Reload: {reload}[/dim]")

    import uvicorn

    uvicorn.run(
        app_factory,
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


if __name__ == "__main__":
    app()

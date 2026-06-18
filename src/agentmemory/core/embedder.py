"""Embedder Protocol.

References:
    - ARCHITECTURE.md §5.3.3 (Embedder Protocol)
"""

from __future__ import annotations

__all__ = ["Embedder"]

from typing import Protocol


class Embedder(Protocol):
    """Protocol for embedding providers.

    All embedder backends (OpenAI, DashScope, local, etc.) must implement this.
    """
    name: str
    dim: int

    async def embed(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...

    async def close(self) -> None:
        """Clean up embedder resources."""
        ...

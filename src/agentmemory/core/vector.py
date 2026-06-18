"""VectorStore ABC.

References:
    - ARCHITECTURE.md §5.3.4 (VectorStore ABC)
"""

from __future__ import annotations

__all__ = ["VectorStore"]

from abc import ABC, abstractmethod

from .types import MemoryItem


class VectorStore(ABC):
    """Abstract base class for vector storage backends.

    All vector store implementations (Qdrant, LanceDB, FAISS, etc.)
    must inherit from this class.
    """

    @abstractmethod
    async def upsert(
        self,
        items: list[MemoryItem],
        vectors: list[list[float]],
    ) -> None:
        """Insert or update memory items with vectors."""
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar vectors, returning (id, score) pairs."""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """Delete items by ID. Returns number of deleted items."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of indexed vectors."""
        ...

    @abstractmethod
    async def rebuild(self) -> None:
        """Rebuild the index (e.g., after bulk deletion)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up storage resources."""
        ...

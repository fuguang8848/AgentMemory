"""MemoryProvider ABC + Memory base class.

References:
    - ARCHITECTURE.md §5.3.1 (MemoryProvider ABC)
    - ARCHITECTURE.md §5.3 (7-verb interface: add/search/get/update/delete/reset/history)
"""

from __future__ import annotations

__all__ = ["MemoryProvider", "Memory"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import MemoryItem, SearchQuery, SearchResult


class MemoryProvider(ABC):
    """7-verb facade for all memory operations.

    All adapters ultimately delegate to this interface.
    """

    @abstractmethod
    async def add(
        self,
        content: str | list[str],
        **kw,
    ) -> list[str]:
        """Add memory content.

        Args:
            content: Single string or list of strings
            **kw: Additional arguments (type, layer, importance, etc.)

        Returns:
            List of created memory IDs
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str | SearchQuery,
        **kw,
    ) -> list[SearchResult]:
        """Search memories.

        Args:
            query: Query string or SearchQuery object
            **kw: Additional search arguments

        Returns:
            List of SearchResult objects
        """
        ...

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryItem | None:
        """Get a single memory item by ID.

        Args:
            memory_id: Memory ID

        Returns:
            MemoryItem or None if not found
        """
        ...

    @abstractmethod
    async def update(
        self,
        memory_id: str,
        **patch,
    ) -> bool:
        """Update a memory item.

        Args:
            memory_id: Memory ID to update
            **patch: Fields to update (content, importance, tags, etc.)

        Returns:
            True if updated, False otherwise
        """
        ...

    @abstractmethod
    async def delete(
        self,
        memory_id: str,
        permanent: bool = False,
    ) -> bool:
        """Delete a memory item.

        Args:
            memory_id: Memory ID to delete
            permanent: If True, permanently delete; otherwise soft-delete

        Returns:
            True if deleted, False otherwise
        """
        ...

    @abstractmethod
    async def reset(self, scope: str = "all") -> int:
        """Reset/delete memories by scope.

        Args:
            scope: Scope of reset ("all", "namespace", "layer", etc.)

        Returns:
            Number of items deleted
        """
        ...

    @abstractmethod
    async def history(
        self,
        memory_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get the change history of a memory item.

        Args:
            memory_id: Memory ID
            limit: Maximum number of history entries

        Returns:
            List of history records as dicts
        """
        ...


class Memory(MemoryProvider):
    """Base Memory class with default implementations.

    Concrete implementations should override as needed.
    """

    pass

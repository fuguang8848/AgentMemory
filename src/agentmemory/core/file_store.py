"""FileStore ABC.

References:
    - ARCHITECTURE.md §5.3.6 (FileStore ABC)
"""

from __future__ import annotations

__all__ = ["FileStore"]

from abc import ABC, abstractmethod

from .types import MemoryItem


class FileStore(ABC):
    """Abstract base class for file-based storage.

    Handles diary entries, memory markdown files, and archive search.
    """

    @abstractmethod
    async def append_diary(
        self,
        date: str,
        entry: str,
        category: str,
    ) -> str:
        """Append a diary entry for a given date.

        Args:
            date: ISO date string (e.g., "2025-01-01")
            entry: Diary content
            category: Category tag

        Returns:
            The diary entry ID.
        """
        ...

    @abstractmethod
    async def write_memory_md(
        self,
        section: str,
        content: str,
    ) -> None:
        """Write content to a section in MEMORY.md.

        Args:
            section: Section name (e.g., "important_facts")
            content: Content to write
        """
        ...

    @abstractmethod
    async def read_diary(self, date: str) -> str:
        """Read diary entries for a given date.

        Args:
            date: ISO date string

        Returns:
            Diary content as string.
        """
        ...

    @abstractmethod
    async def list_diaries(
        self,
        from_date: str,
        to_date: str,
    ) -> list[str]:
        """List diary dates within a date range.

        Args:
            from_date: Start date (ISO)
            to_date: End date (ISO)

        Returns:
            List of date strings.
        """
        ...

    @abstractmethod
    async def search_archive(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        """Search the file archive.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of matching MemoryItems.
        """
        ...

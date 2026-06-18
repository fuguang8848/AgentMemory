"""FactExtractor Protocol.

References:
    - ARCHITECTURE.md §5 (fact extraction step)
"""

from __future__ import annotations

__all__ = ["FactExtractor"]

from typing import Protocol

from .types import MemoryItem


class FactExtractor(Protocol):
    """Protocol for fact extraction from raw content.

    Extractors transform unstructured or semi-structured content
    into structured MemoryItems.
    """

    name: str

    async def extract(
        self,
        content: str | list[dict],
        *,
        source: str = "inference",
        **kw,
    ) -> list[MemoryItem]:
        """Extract facts from content.

        Args:
            content: Raw text or message list
            source: Source identifier (user/system/inference/reflection)
            **kw: Additional extractor-specific arguments

        Returns:
            List of extracted MemoryItems
        """
        ...

    async def extract_one(
        self,
        content: str,
        **kw,
    ) -> MemoryItem | None:
        """Extract a single fact from content.

        Args:
            content: Raw text
            **kw: Additional arguments

        Returns:
            A single MemoryItem or None if extraction fails
        """
        ...

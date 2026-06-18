"""Retriever ABC.

References:
    - ARCHITECTURE.md §5.3.8 (Retriever ABC)
"""

from __future__ import annotations

__all__ = ["Retriever", "RetrievalStrategy"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import SearchQuery, SearchResult


class RetrievalStrategy:
    """Base class for retrieval strategies."""

    name: str

    async def retrieve(
        self,
        query: SearchQuery,
        ctx: dict[str, Any],
    ) -> list[SearchResult]:
        """Execute the retrieval strategy.

        Args:
            query: SearchQuery object
            ctx: Execution context

        Returns:
            List of SearchResult objects
        """
        raise NotImplementedError


class Retriever(ABC):
    """Abstract base class for multi-strategy retrievers.

    Coordinates multiple retrieval strategies (vector, BM25, graph, importance).
    """

    @abstractmethod
    def add_strategy(
        self,
        name: str,
        strategy: RetrievalStrategy,
    ) -> None:
        """Register a retrieval strategy by name."""
        ...

    @abstractmethod
    async def retrieve(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Execute all registered strategies and merge results.

        Args:
            query: SearchQuery object

        Returns:
            Merged list of SearchResult objects
        """
        ...

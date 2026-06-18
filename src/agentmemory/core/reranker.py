"""Reranker Protocol.

References:
    - ARCHITECTURE.md §5 (rerank step in hybrid retrieval)
"""

from __future__ import annotations

__all__ = ["Reranker"]

from typing import Protocol

from .types import SearchResult


class Reranker(Protocol):
    """Protocol for result reranking.

    Rerankers take initial search results and reorder them
    using additional signals (e.g., cross-encoder, LLM-based).
    """

    name: str

    async def rerank(
        self,
        results: list[SearchResult],
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results.

        Args:
            results: Initial search results
            query: Original query string
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of SearchResult objects
        """
        ...

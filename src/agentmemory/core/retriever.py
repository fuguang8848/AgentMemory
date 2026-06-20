"""Retriever ABC — 多策略检索抽象基类

本模块定义检索策略的抽象接口。所有具体检索器（如 VectorRetriever、
BM25Retriever、GraphRetriever 等）需继承 Retriever 并实现对应抽象方法。

架构参考：
    - ARCHITECTURE.md §5.3.8 (Retriever ABC)
    - LangGraph Checkpointing 模式（状态检查点 → 检索结果合并）

当前状态：框架定义阶段，具体实现尚未接入。
NotImplementedError 在抽象方法上是合理的设计选择。

Usage:
    class MyRetriever(Retriever):
        def add_strategy(self, name: str, strategy: RetrievalStrategy) -> None:
            ...

        async def retrieve(self, query: SearchQuery) -> list[SearchResult]:
            ...
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

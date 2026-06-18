"""
Identity Reranker (No-Op)
M1 Default Reranker
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Reranked result"""
    id: str
    score: float
    content: Any


class Reranker:
    """Protocol for reranking providers"""

    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: int | None = None,
        **kwargs
    ) -> list[RerankResult]:
        """Rerank documents based on query"""
        raise NotImplementedError


class IdentityReranker(Reranker):
    """
    Identity Reranker - returns documents as-is without reordering.
    M1 Default Reranker (no-op / pass-through).
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: int | None = None,
        **kwargs
    ) -> list[RerankResult]:
        """
        Identity reranking - returns documents in original order.
        Acts as a pass-through/identity function.

        Args:
            query: The query string (ignored)
            documents: List of documents to rerank
            top_k: Number of top results to return (default: all)

        Returns:
            List of RerankResult in original order
        """
        if top_k is None:
            top_k = len(documents)

        results = []
        for i, doc in enumerate(documents[:top_k]):
            # Extract id and content from document
            if isinstance(doc, dict):
                doc_id = doc.get("id", str(i))
                content = doc.get("content", doc.get("text", doc))
                score = doc.get("score", 1.0 - (i / len(documents)))
            else:
                doc_id = str(i)
                content = doc
                score = 1.0 - (i / len(documents))

            results.append(RerankResult(
                id=doc_id,
                score=score,
                content=content
            ))

        return results

"""Retrieve Pipeline - hybrid retrieval with vector + BM25 + importance weighting.

References:
    - ARCHITECTURE.md §10.5 (lines 1570-1595)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.types import MemoryItem, SearchQuery, SearchResult
    from ..core.vector import VectorStore
    from ..core.embedder import Embedder
    from ..core.retriever import RetrievalStrategy

_DEFAULT_WEIGHTS = {"vector": 0.6, "bm25": 0.3, "importance": 0.1}
_DEFAULT_TOP_K = 50  # Broad recall phase


class BM25Simple:
    """Simple in-memory BM25 implementation.

    For production, replace with rank_bm25 or similar library.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: dict[str, str] = {}  # id -> content
        self._index: dict[str, dict[str, int]] = {}  # id -> {word: count}
        self._doc_lengths: dict[str, float] = {}
        self._avg_doc_length = 0.0
        self._idf: dict[str, float] = {}
        self._total_docs = 0

    def index(self, items: list[MemoryItem]) -> None:
        """Build BM25 index from memory items."""
        import math
        import re

        self._documents.clear()
        self._index.clear()
        self._doc_lengths.clear()

        for item in items:
            self._documents[item.id] = item.content
            # Tokenize
            tokens = re.findall(r"\w+", item.content.lower())
            self._index[item.id] = {}
            for token in tokens:
                self._index[item.id][token] = self._index[item.id].get(token, 0) + 1
            self._doc_lengths[item.id] = len(tokens)

        self._total_docs = len(self._documents)
        if self._total_docs == 0:
            return

        self._avg_doc_length = sum(self._doc_lengths.values()) / self._total_docs

        # Compute IDF
        df: dict[str, int] = {}
        for idx in self._index.values():
            for word in idx:
                df[word] = df.get(word, 0) + 1

        for word, doc_freq in df.items():
            self._idf[word] = math.log((self._total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Search BM25 index.

        Returns:
            List of (item_id, score) tuples
        """
        import re

        if self._total_docs == 0:
            return []

        query_tokens = re.findall(r"\w+", query.lower())
        if not query_tokens:
            return []

        scores: dict[str, float] = {}

        for doc_id in self._documents:
            score = 0.0
            doc_len = self._doc_lengths[doc_id]
            doc_index = self._index[doc_id]

            for term in query_tokens:
                if term not in doc_index:
                    continue

                tf = doc_index[term]
                idf = self._idf.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_length)
                score += idf * numerator / denominator

            if score > 0:
                scores[doc_id] = score

        # Sort and return top-k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]


class RetrievePipeline:
    """Hybrid retrieval pipeline: vector + BM25 + importance weighted.

    Two-phase retrieval:
    1. Broad recall: vector top-50 + BM25 top-50 (configurable)
    2. Weighted merge with configurable weights
    3. Optional rerank
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        items: list[MemoryItem] | None = None,
        weights: dict[str, float] | None = None,
        top_k_candidates: int = _DEFAULT_TOP_K,
    ):
        """Initialize RetrievePipeline.

        Args:
            vector_store: L3 VectorStore instance
            embedder: Embedder for query vectorization
            items: Optional in-memory list of MemoryItem (for BM25)
            weights: Retrieval weights dict, default {"vector": 0.6, "bm25": 0.3, "importance": 0.1}
            top_k_candidates: Number of candidates per strategy
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.items = items or []
        self.weights = weights or _DEFAULT_WEIGHTS.copy()
        self.top_k_candidates = top_k_candidates

        # Build BM25 index if items provided
        self._bm25 = BM25Simple()
        if self.items:
            self._bm25.index(self.items)

    def update_items(self, items: list[MemoryItem]) -> None:
        """Update the item list and rebuild BM25 index."""
        self.items = items
        self._bm25.index(items)

    async def retrieve(self, query: SearchQuery) -> list[SearchResult]:
        """Execute hybrid retrieval.

        Args:
            query: SearchQuery with text, top_k, filters, etc.

        Returns:
            List of SearchResult sorted by weighted score
        """
        # Phase 1: Broad recall from multiple strategies
        vector_results: list[tuple[str, float]] = []
        bm25_results: list[tuple[str, float]] = []

        # Vector search
        try:
            query_vector = await self.embedder.embed_query(query.text)
            filters = self._build_filter(query)
            vector_results = await self.vector_store.search(
                query_vector, top_k=self.top_k_candidates, filter=filters
            )
        except Exception:
            pass

        # BM25 search
        if self._bm25._total_docs > 0:
            bm25_results = self._bm25.search(query.text, top_k=self.top_k_candidates)

        # Importance-based results (from in-memory items)
        importance_results = self._importance_search(query)

        # Phase 2: Merge and deduplicate
        merged = self._merge_results(vector_results, bm25_results, importance_results, query)

        # Phase 3: Apply filters and limits
        filtered = self._apply_filters(merged, query)

        # Trim to top_k
        return filtered[: query.top_k]

    def _merge_results(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        importance_results: list[tuple[str, float]],
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Merge results from multiple strategies with weighted scoring.

        Returns SearchResult list with explanations.
        """
        from ..core.types import MemoryLayer, SearchResult

        # Normalize scores to 0-1 range
        def normalize(scores: list[tuple[str, float]]) -> dict[str, float]:
            if not scores:
                return {}
            max_score = max(s for _, s in scores)
            min_score = min(s for _, s in scores)
            range_score = max_score - min_score if max_score != min_score else 1.0
            return {id_: (s - min_score) / range_score for id_, s in scores}

        vec_norm = normalize(vector_results)
        bm25_norm = normalize(bm25_results)
        imp_norm = normalize(importance_results)

        # Collect all item IDs
        all_ids: set[str] = set()
        for id_, _ in vector_results:
            all_ids.add(id_)
        for id_, _ in bm25_results:
            all_ids.add(id_)
        for id_, _ in importance_results:
            all_ids.add(id_)

        # Calculate weighted scores
        results: list[SearchResult] = []
        w_vec = self.weights.get("vector", 0.6)
        w_bm25 = self.weights.get("bm25", 0.3)
        w_imp = self.weights.get("importance", 0.1)

        for item_id in all_ids:
            # Find the MemoryItem
            item = next((i for i in self.items if i.id == item_id), None)
            if item is None:
                continue

            v_score = vec_norm.get(item_id, 0.0)
            b_score = bm25_norm.get(item_id, 0.0)
            i_score = imp_norm.get(item_id, 0.0)

            total_score = w_vec * v_score + w_bm25 * b_score + w_imp * i_score

            # Determine sources
            sources = []
            if item_id in [id_ for id_, _ in vector_results]:
                sources.append("vector")
            if item_id in [id_ for id_, _ in bm25_results]:
                sources.append("bm25")
            if item_id in [id_ for id_, _ in importance_results]:
                sources.append("importance")

            result = SearchResult(
                item=item,
                score=total_score,
                layer=item.layer,
                sources=sources,
                explanation={"vector": v_score, "bm25": b_score, "importance": i_score},
            )
            results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _importance_search(self, query: SearchQuery) -> list[tuple[str, float]]:
        """Simple importance-based retrieval.

        Returns items sorted by importance score.
        """
        if not self.items:
            return []

        scored = [(item.id, item.importance) for item in self.items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k_candidates]

    def _build_filter(self, query: SearchQuery) -> dict | None:
        """Build filter dict for vector store search."""
        filters: dict[str, Any] = {}

        if query.filter_type:
            filters["type"] = [t.value for t in query.filter_type]
        if query.filter_layer:
            filters["layer"] = [l.value for l in query.filter_layer]
        if query.filter_tags:
            filters["tags"] = {"$in": query.filter_tags}
        if query.tenant_id:
            filters["tenant_id"] = query.tenant_id
        if query.namespace:
            filters["namespace"] = query.namespace

        return filters if filters else None

    def _apply_filters(self, results: list[SearchResult], query: SearchQuery) -> list[SearchResult]:
        """Apply post-retrieval filters."""
        filtered = []

        for result in results:
            # Min score filter
            if result.score < query.min_score:
                continue

            # Type filter
            if query.filter_type and result.item.type not in query.filter_type:
                continue

            # Layer filter
            if query.filter_layer and result.item.layer not in query.filter_layer:
                continue

            # Tags filter
            if query.filter_tags:
                if not any(tag in result.item.tags for tag in query.filter_tags):
                    continue

            filtered.append(result)

        return filtered

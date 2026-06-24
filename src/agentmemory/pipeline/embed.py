"""Embed Pipeline - batch vectorization with embedding cache.

References:
    - ARCHITECTURE.md §10.3 (lines 1534-1551)
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.types import MemoryItem
    from ..core.embedder import Embedder

_DEFAULT_BATCH_SIZE = 32
_CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days


class EmbedCache:
    """Simple in-memory embedding cache with TTL.

    Cache key: (text_hash, embedder_model) -> vector
    
    Plato's Cave Issue: Cache is a "shadow" of real embeddings - if source data changes,
    cached embeddings become stale shadows. Fixed 7-day TTL without refresh mechanism
    means stale shadows persist until expiry.
    
   芒格工程学 Issue: Unbounded cache growth → memory leak. No cleanup of expired entries
    except on access (get). Should use max_size + LRU eviction.
    """

    # 芒格工程学: Safety margin - max cache size to prevent memory exhaustion
    _MAX_CACHE_SIZE = 10000

    def __init__(self, ttl: int = _CACHE_TTL_SECONDS, max_size: int = _MAX_CACHE_SIZE):
        self._cache: dict[str, tuple[list[float], float]] = {}  # key -> (vector, expiry)
        self._ttl = ttl
        self._max_size = max_size
        self._lock = asyncio.Lock()

    def _make_key(self, text: str, model: str) -> str:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{model}:{text_hash}"

    async def get(self, text: str, model: str) -> list[float] | None:
        """Get cached embedding if not expired."""
        key = self._make_key(text, model)
        async with self._lock:
            if key not in self._cache:
                return None
            vector, expiry = self._cache[key]
            if time.time() > expiry:
                del self._cache[key]
                return None
            return vector

    async def set(self, text: str, model: str, vector: list[float]) -> None:
        """Cache an embedding with TTL.
        
        芒格工程学 Fix: Evict oldest entries when cache is full (近似LRU).
        This prevents unbounded memory growth (memory leak).
        """
        key = self._make_key(text, model)
        expiry = time.time() + self._ttl
        async with self._lock:
            # 芒格工程学: If cache full, evict ~20% oldest entries to make space
            if len(self._cache) >= self._max_size:
                self._evict_oldest(int(self._max_size * 0.2))
            self._cache[key] = (vector, expiry)
    
    def _evict_oldest(self, count: int) -> None:
        """Evict oldest 'count' entries from cache (approximate LRU)."""
        if not self._cache:
            return
        # Sort by expiry (oldest first) and remove
        sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])
        for key in sorted_keys[:count]:
            del self._cache[key]
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
    
    async def size(self) -> int:
        """Return current cache size."""
        async with self._lock:
            return len(self._cache)

    async def get_batch(
        self, texts: list[str], model: str
    ) -> tuple[list[list[float]], list[int]]:
        """Get batch of cached embeddings.

        Returns:
            Tuple of (cached_vectors, missing_indices)
            cached_vectors[i] corresponds to texts[missing_indices[i]]
        """
        cached = []
        missing_idx = []

        for i, text in enumerate(texts):
            vec = await self.get(text, model)
            if vec is not None:
                cached.append((i, vec))
            else:
                missing_idx.append(i)

        return [v for _, v in cached], missing_idx


class EmbedPipeline:
    """Embedding pipeline with batching and caching.

    Input: list of MemoryItem
    Output: MemoryItem list with embeddings filled in
    """

    def __init__(
        self,
        embedder: Embedder,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        cache: EmbedCache | None = None,
        fallback_embedder: Embedder | None = None,
    ):
        """Initialize EmbedPipeline.

        Args:
            embedder: Primary Embedder instance
            batch_size: Batch size for embedding requests
            cache: Optional EmbedCache instance (shared across calls)
            fallback_embedder: Optional fallback embedder if primary fails
        """
        self.embedder = embedder
        self.batch_size = batch_size
        self.cache = cache or EmbedCache()
        self.fallback_embedder = fallback_embedder

    async def run(self, items: list[MemoryItem]) -> list[MemoryItem]:
        """Run embedding pipeline on items.

        Args:
            items: List of MemoryItem to embed

        Returns:
            MemoryItem list with embeddings populated
        """
        if not items:
            return items

        # Collect texts
        texts = [item.content for item in items]
        model_name = self.embedder.name

        # Check cache for all texts
        cached_vectors, missing_idx = await self.cache.get_batch(texts, model_name)

        # Build result list
        result = [None] * len(items)
        cached_map: dict[int, list[float]] = {}

        # Fill in cached vectors
        cached_count = 0
        for i, text in enumerate(texts):
            vec = await self.cache.get(text, model_name)
            if vec is not None:
                result[i] = vec
                cached_map[i] = vec
                cached_count += 1

        # Batch embed missing items
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]

            try:
                new_vectors = await self._embed_batch_with_fallback(missing_texts)
            except Exception:
                # On complete failure, return items without embeddings
                for i in missing_idx:
                    items[i].embedding = None
                return items

            # Cache new vectors and assign to results
            for j, idx in enumerate(missing_idx):
                vector = new_vectors[j]
                result[idx] = vector
                # Store in cache
                await self.cache.set(texts[idx], model_name, vector)

        # Assign embeddings to items
        for i, item in enumerate(items):
            item.embedding = result[i]

        return items

    async def _embed_batch_with_fallback(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch with fallback on failure.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            return await self.embedder.embed(texts, batch_size=self.batch_size)
        except Exception:
            if self.fallback_embedder is not None:
                return await self.fallback_embedder.embed(texts, batch_size=self.batch_size)
            raise

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache first
        model_name = self.embedder.name
        cached = await self.cache.get(text, model_name)
        if cached is not None:
            return cached

        # Embed and cache
        vectors = await self._embed_batch_with_fallback([text])
        vector = vectors[0]
        await self.cache.set(text, model_name, vector)
        return vector

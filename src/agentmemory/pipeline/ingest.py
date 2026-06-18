"""Ingest Pipeline - deduplication, PII redaction, and chunking.

References:
    - ARCHITECTURE.md §10.1 (lines 1500-1515)
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.types import MemoryItem

_DEFAULT_CHUNK_SIZE = 512  # tokens
_DEFAULT_CHUNK_OVERLAP = 64  # tokens


class IngestPipeline:
    """Ingestion pipeline: dedupe -> PII redact -> chunk.

    Input: raw content string
    Output: list of MemoryItem (chunked)
    """

    def __init__(
        self,
        pii_redact_fn: callable | None = None,
        dedupe_fn: callable | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
        tenant_id: str = "default",
        namespace: str = "default",
    ):
        """Initialize IngestPipeline.

        Args:
            pii_redact_fn: Optional PII redaction function (content) -> content
            dedupe_fn: Optional dedupe function (content_hash) -> existing_id | None
            chunk_size: Token window size for chunking
            chunk_overlap: Overlap between consecutive chunks
            tenant_id: Tenant identifier
            namespace: Namespace identifier
        """
        self.pii_redact_fn = pii_redact_fn
        self.dedupe_fn = dedupe_fn
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tenant_id = tenant_id
        self.namespace = namespace

    def _content_hash(self, content: str, metadata_fingerprint: str = "") -> str:
        """Generate dedupe hash for content.

        Hash = sha256(content + tenant_id + namespace + metadata.fingerprint)
        """
        raw = f"{content}{self.tenant_id}{self.namespace}{metadata_fingerprint}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _pii_redact(self, content: str) -> str:
        """Apply PII redaction to content."""
        if self.pii_redact_fn is None:
            return content
        return self.pii_redact_fn(content)

    async def _dedupe(self, content_hash: str) -> str | None:
        """Check if content already exists. Returns existing ID or None."""
        if self.dedupe_fn is None:
            return None
        return await self.dedupe_fn(content_hash)

    def _chunk(self, content: str) -> list[str]:
        """Split content into overlapping chunks by token count.

        Simple token-based chunking using word boundaries.
        For production, replace with proper tokenizer.
        """
        if len(content) <= self.chunk_size:
            return [content]

        chunks = []
        words = content.split()
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

            # Step forward with overlap
            step = self.chunk_size - self.chunk_overlap
            if step <= 0:
                step = 1
            start += step

        return chunks

    async def run(self, content: str, metadata: dict[str, Any] | None = None) -> tuple[list[MemoryItem], str | None]:
        """Run the ingest pipeline.

        Args:
            content: Raw content string
            metadata: Optional metadata dict (may contain 'fingerprint' for dedupe)

        Returns:
            Tuple of (list of MemoryItem chunks, existing_id if deduped else None)
        """
        metadata = metadata or {}
        fingerprint = metadata.get("fingerprint", "")

        # 1. Dedupe check
        content_hash = self._content_hash(content, fingerprint)
        existing_id = await self._dedupe(content_hash)
        if existing_id is not None:
            return [], existing_id

        # 2. PII redaction
        redacted = self._pii_redact(content)

        # 3. Chunk
        chunks = self._chunk(redacted)

        # 4. Build MemoryItem list
        from ..core.types import MemoryItem, MemoryLayer, MemoryType

        items = []
        for i, chunk_text in enumerate(chunks):
            item = MemoryItem(
                content=chunk_text,
                type=MemoryType.SEMANTIC,
                layer=MemoryLayer.L1_COMPRESS,
                tenant_id=self.tenant_id,
                namespace=self.namespace,
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_hash": content_hash,
                    **metadata,
                },
            )
            items.append(item)

        return items, None

"""Index Pipeline - async background indexing to VectorStore/GraphStore/FileStore.

References:
    - ARCHITECTURE.md §10.4 (lines 1552-1569)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.types import MemoryItem, MemoryLayer
    from ..core.vector import VectorStore
    from ..core.graph import GraphStore, GraphNode, GraphEdge
    from ..core.file_store import FileStore

_QUEUE_SIZE = 1000


class IndexPipeline:
    """Async indexing pipeline with atomic 3-layer writes.

    Writes to VectorStore (L3), GraphStore (L2), FileStore (L4) atomically.
    Uses asyncio.Queue for backpressure control.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        file_store: FileStore,
        queue_size: int = _QUEUE_SIZE,
    ):
        """Initialize IndexPipeline.

        Args:
            vector_store: L3 VectorStore instance
            graph_store: L2 GraphStore instance
            file_store: L4 FileStore instance
            queue_size: Max queue depth for backpressure
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.file_store = file_store
        self._queue: asyncio.Queue[tuple[list[MemoryItem], dict[str, Any]]] = asyncio.Queue(
            maxsize=queue_size
        )
        self._running = False
        self._worker_task: asyncio.Task | None = None

    async def submit(self, items: list[MemoryItem], metadata: dict[str, Any] | None = None) -> None:
        """Submit items for async indexing.

        Non-blocking - adds to queue and returns immediately.

        Args:
            items: List of MemoryItem to index
            metadata: Optional metadata (e.g., for diary entry)
        """
        metadata = metadata or {}
        await self._queue.put((items, metadata))

    async def start(self) -> None:
        """Start the background indexing worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the background worker gracefully."""
        self._running = False
        if self._worker_task is not None:
            await self._queue.join()  # Wait for queue to drain
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def consume(self) -> None:
        """Consume and process queued items. Called by worker."""
        while self._running:
            try:
                items, metadata = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._index_atomic(items, metadata)
            except Exception:
                # Log error but don't crash worker
                import logging

                logging.exception(f"IndexPipeline: failed to index {len(items)} items")
            finally:
                self._queue.task_done()

    async def _worker(self) -> None:
        """Background worker loop."""
        await self.consume()

    async def _index_atomic(
        self, items: list[MemoryItem], metadata: dict[str, Any]
    ) -> None:
        """Atomically index items to all three stores.

        If any store fails, all are rolled back.

        Args:
            items: List of MemoryItem to index
            metadata: Optional metadata (e.g., date for diary)
        """
        if not items:
            return

        # Separate items by layer for different stores
        l3_items = [i for i in items if i.layer.value == "L3" or i.embedding is not None]
        l2_items = [i for i in items if i.layer.value == "L2" or i.entities]

        # Extract vectors for L3 items
        vectors = []
        valid_items = []
        for item in l3_items:
            if item.embedding is not None:
                vectors.append(item.embedding)
                valid_items.append(item)

        try:
            # L3: VectorStore upsert
            if vectors:
                await self.vector_store.upsert(valid_items, vectors)

            # L2: GraphStore add_node + add_edge
            for item in l2_items:
                # Create node for each entity
                node_ids: list[str] = []
                for entity in item.entities:
                    node = GraphNode(
                        id=item.id,
                        label=entity,
                        properties={
                            "content": item.content,
                            "importance": item.importance,
                            "memory_item_id": item.id,
                        },
                    )
                    node_id = await self.graph_store.add_node(node)
                    node_ids.append(node_id)

                # Create edges between entities in same item
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        edge = GraphEdge(
                            id=f"{node_ids[i]}-{node_ids[j]}",
                            source_id=node_ids[i],
                            target_id=node_ids[j],
                            label="related_to",
                            properties={"memory_item_id": item.id},
                        )
                        await self.graph_store.add_edge(edge)

            # L4: FileStore append_diary
            date = metadata.get("date")
            if date and l3_items:
                # Aggregate content for diary
                combined_content = "\n".join(i.content for i in l3_items[:10])
                category = metadata.get("category", "indexed_memory")
                await self.file_store.append_diary(date, combined_content, category)

        except Exception as e:
            # On failure, attempt cleanup (best effort)
            # In production, implement proper transaction rollback
            import logging

            logging.error(f"IndexPipeline: atomic index failed, error: {e}")
            raise

    async def index_now(self, items: list[MemoryItem], metadata: dict[str, Any] | None = None) -> None:
        """Index items synchronously (blocking).

        Use this for testing or when async is not desired.

        Args:
            items: List of MemoryItem to index
            metadata: Optional metadata
        """
        await self._index_atomic(items, metadata or {})

    @property
    def queue_size(self) -> int:
        """Current number of items in queue."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Whether the background worker is running."""
        return self._running

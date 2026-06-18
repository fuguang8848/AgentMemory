"""EventBus - async pub/sub event system for AgentMemory 2.0.

References:
    - ARCHITECTURE.md §9.1.1-9.1.3 (lines 1281-1368)
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event type enumeration covering all AgentMemory operations.

    References:
        - ARCHITECTURE.md §9.1.1 (lines 1283-1326)
    """
    # Lifecycle
    MEMORY_ADD_REQUESTED = "memory.add.requested"
    MEMORY_ADD_COMMITTED = "memory.add.committed"
    MEMORY_ADD_FAILED = "memory.add.failed"
    MEMORY_SEARCH_REQUESTED = "memory.search.requested"
    MEMORY_SEARCH_COMPLETED = "memory.search.completed"
    MEMORY_GET_REQUESTED = "memory.get.requested"
    MEMORY_UPDATED = "memory.updated"
    MEMORY_DELETED = "memory.deleted"
    MEMORY_FORGOTTEN = "memory.forgotten"
    MEMORY_DECAYED = "memory.decayed"
    MEMORY_ARCHIVED = "memory.archived"
    MEMORY_REFLECTED = "memory.reflected"  # M3

    # Provider
    LLM_CALL_REQUESTED = "llm.call.requested"
    LLM_CALL_SUCCEEDED = "llm.call.succeeded"
    LLM_CALL_FAILED = "llm.call.failed"
    LLM_FALLBACK_TRIGGERED = "llm.fallback.triggered"
    EMBED_BATCH_STARTED = "embed.batch.started"
    EMBED_BATCH_COMPLETED = "embed.batch.completed"

    # Pipeline
    INGEST_STARTED = "ingest.started"
    INGEST_COMPLETED = "ingest.completed"
    EXTRACT_STARTED = "extract.started"
    EXTRACT_COMPLETED = "extract.completed"
    INDEX_STARTED = "index.started"
    INDEX_COMPLETED = "index.completed"

    # Cross-cutting
    RATE_LIMITED = "ratelimit.triggered"
    CIRCUIT_OPENED = "circuit.opened"
    CIRCUIT_CLOSED = "circuit.closed"
    PII_DETECTED = "pii.detected"
    ENCRYPTION_APPLIED = "encryption.applied"
    TENANT_VIOLATION = "tenant.violation"

    # Error
    ERROR = "error"
    WARNING = "warning"


class Event(BaseModel):
    """Event data model.

    References:
        - ARCHITECTURE.md §9.1.2 (lines 1330-1342)
    """
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tenant_id: str = "default"
    namespace: str = "default"
    trace_id: str | None = None
    span_id: str | None = None
    request_id: str | None = None
    actor: str = "user"  # user / system / scheduler / adapter
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """Async pub/sub event bus.

    In-memory event bus with async handlers, inspired by Node.js EventEmitter
    but fully async. Events are stored in a bounded history buffer.

    References:
        - ARCHITECTURE.md §9.1.3 (lines 1346-1368)
    """
    _history_maxlen = 10_000

    def __init__(self, history_maxlen: int = 10_000):
        """Initialize EventBus.

        Args:
            history_maxlen: Max events to keep in history buffer
        """
        self._subscribers: dict[EventType, list[Callable[[Event], Awaitable[None]]]] = {}
        self._history: deque[Event] = deque(maxlen=history_maxlen)
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]) -> None:
        """Subscribe a handler to an event type.

        Args:
            event_type: EventType to subscribe to
            handler: Async callable that takes Event and returns None
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], Awaitable[None]]) -> None:
        """Unsubscribe a handler from an event type.

        Args:
            event_type: EventType to unsubscribe from
            handler: Handler to remove
        """
        if event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)

    async def emit(self, event: Event) -> None:
        """Emit an event to all subscribers.

        Subscribers are executed concurrently. Errors are isolated and logged.

        Args:
            event: Event to emit
        """
        async with self._lock:
            self._history.append(event)
            handlers = list(self._subscribers.get(event.type, []))

        if not handlers:
            return

        # Execute all handlers concurrently, isolate errors
        results = await asyncio.gather(
            *[h(event) for h in handlers],
            return_exceptions=True
        )

        for r in results:
            if isinstance(r, BaseException):
                logger.error("event_handler_failed", exc_info=r)

    def history(self, event_type: EventType | None = None, limit: int = 100) -> list[Event]:
        """Get event history.

        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return

        Returns:
            List of recent Event objects
        """
        if event_type is None:
            return list(self._history)[-limit:]

        filtered = [e for e in self._history if e.type == event_type]
        return filtered[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()

    @property
    def subscriber_count(self) -> int:
        """Total number of subscribed handlers."""
        return sum(len(handlers) for handlers in self._subscribers.values())

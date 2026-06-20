"""
Multi-Agent Memory Bus — Cross-Agent Shared Memory with Namespace Isolation

SpectrAI Enhancement: v0.4 §4.2
- MemoryBus: 跨Agent共享内存读写，支持 namespace 隔离
- 权限控制：基于 permissions.py 的读写权限
- 消息订阅：Agent可订阅特定memory变化事件
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable, Literal

import aiofiles

from .permissions import PermissionEngine, PermissionContext


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class MemoryMessage:
    """Memory change event message"""
    id: str
    namespace: str
    agent_id: str
    operation: Literal["read", "write", "delete"]
    memory_id: str
    content: str | None = None
    metadata: dict = field(default_factory=dict)
    timestamp: str = ""
    trace_id: str | None = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class SharedMemoryEntry:
    """Shared memory entry stored in the bus"""
    id: str
    namespace: str
    content: str
    metadata: dict = field(default_factory=dict)
    owner_agent_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    version: int = 1

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class Subscription:
    """Agent subscription to memory events"""
    id: str
    agent_id: str
    namespace: str
    filter_pattern: str | None = None  # Glob pattern for memory_id filtering
    event_types: list[str] = field(default_factory=list)  # read/write/delete


# ============================================================================
# Memory Bus Implementation
# ============================================================================


class MemoryBus:
    """
    Multi-Agent Memory Bus with namespace isolation and permission control.
    
    Features:
    - Namespace isolation: agents can only access memories in their namespaces
    - Permission-based access control via PermissionEngine
    - Message subscription: agents can subscribe to specific memory change events
    - Async pub/sub for cross-agent communication
    
    Usage:
        bus = MemoryBus(storage_path="/path/to/memory", permission_engine=perm_engine)
        await bus.register_agent("agent_1", allowed_paths=["A.项目/**"])
        await bus.write("agent_1", "A.项目/石榴籽", "memory_id", "content")
        await bus.subscribe("agent_1", "A.项目/**", event_types=["write"])
    """

    BUS_FILE: str = ".memory_bus.json"
    SCHEMA_VERSION: int = 1

    def __init__(
        self,
        storage_path: Path | str,
        permission_engine: PermissionEngine | None = None,
    ):
        """
        Initialize MemoryBus.
        
        Args:
            storage_path: Base path for memory storage
            permission_engine: Optional PermissionEngine for access control
        """
        self.storage_path = Path(storage_path)
        self._bus_file = self.storage_path / self.BUS_FILE
        self._permission_engine = permission_engine or PermissionEngine(storage_path)
        
        # In-memory cache
        self._memories: dict[str, dict[str, SharedMemoryEntry]] = {}  # namespace -> memory_id -> entry
        self._namespaces: set[str] = {"default"}
        
        # Subscriptions: namespace -> list[Subscription]
        self._subscriptions: dict[str, list[Subscription]] = {}
        
        # Message queue for async delivery
        self._message_queue: asyncio.Queue[MemoryMessage] = asyncio.Queue()
        self._lock = asyncio.Lock()
        
        # Event handlers storage
        self._event_handlers: dict[str, list[Callable[[MemoryMessage], Awaitable[None]]]] = {}

    # =========================================================================
    # Agent Registration
    # =========================================================================

    async def register_agent(
        self,
        agent_id: str,
        allowed_paths: list[str] | None = None,
        denied_paths: list[str] | None = None,
        read_only_paths: list[str] | None = None,
    ) -> dict:
        """
        Register an agent with the memory bus.
        
        Args:
            agent_id: Unique agent identifier
            allowed_paths: Paths the agent can access
            denied_paths: Paths explicitly denied
            read_only_paths: Paths the agent can only read
            
        Returns:
            Registration confirmation
        """
        await self._permission_engine.register(
            agent_id=agent_id,
            allowed_paths=allowed_paths or [],
            denied_paths=denied_paths or [],
            read_only_paths=read_only_paths or [],
        )
        return {"status": "registered", "agent_id": agent_id}

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent and clean up subscriptions."""
        result = await self._permission_engine.unregister(agent_id)
        
        # Remove agent's subscriptions
        async with self._lock:
            for namespace in self._subscriptions:
                self._subscriptions[namespace] = [
                    s for s in self._subscriptions[namespace] if s.agent_id != agent_id
                ]
        
        return result

    # =========================================================================
    # Namespace Management
    # =========================================================================

    async def create_namespace(self, namespace: str) -> None:
        """Create a new namespace."""
        async with self._lock:
            self._namespaces.add(namespace)
            if namespace not in self._memories:
                self._memories[namespace] = {}
            if namespace not in self._subscriptions:
                self._subscriptions[namespace] = []

    async def list_namespaces(self, agent_id: str) -> list[str]:
        """List namespaces accessible to an agent."""
        return list(self._namespaces)

    # =========================================================================
    # Memory Read/Write Operations
    # =========================================================================

    async def write(
        self,
        agent_id: str,
        namespace: str,
        memory_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> MemoryMessage | None:
        """
        Write memory to the bus with permission check.
        
        Args:
            agent_id: Writing agent
            namespace: Target namespace
            memory_id: Memory identifier
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            MemoryMessage if successful, None if permission denied
        """
        # Check permission
        target_path = f"{namespace}/{memory_id}"
        perm_ctx = await self._permission_engine.check(agent_id, "write", target_path)
        
        if not perm_ctx.granted:
            return None
        
        # Ensure namespace exists
        if namespace not in self._namespaces:
            await self.create_namespace(namespace)
        
        entry = SharedMemoryEntry(
            id=memory_id,
            namespace=namespace,
            content=content,
            metadata=metadata or {},
            owner_agent_id=agent_id,
        )
        
        async with self._lock:
            if namespace not in self._memories:
                self._memories[namespace] = {}
            
            existing = self._memories[namespace].get(memory_id)
            if existing:
                entry.created_at = existing.created_at
                entry.version = existing.version + 1
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            
            self._memories[namespace][memory_id] = entry
        
        # Create and emit message
        message = MemoryMessage(
            id=str(uuid.uuid4()),
            namespace=namespace,
            agent_id=agent_id,
            operation="write",
            memory_id=memory_id,
            content=content,
            metadata=metadata or {},
        )
        
        await self._emit_message(message)
        await self._persist()
        
        return message

    async def read(
        self,
        agent_id: str,
        namespace: str,
        memory_id: str,
    ) -> SharedMemoryEntry | None:
        """
        Read memory from the bus with permission check.
        
        Args:
            agent_id: Reading agent
            namespace: Source namespace
            memory_id: Memory identifier
            
        Returns:
            SharedMemoryEntry if found and permitted, None otherwise
        """
        # Check permission
        target_path = f"{namespace}/{memory_id}"
        perm_ctx = await self._permission_engine.check(agent_id, "read", target_path)
        
        if not perm_ctx.granted:
            return None
        
        async with self._lock:
            if namespace not in self._memories:
                return None
            entry = self._memories[namespace].get(memory_id)
        
        if entry:
            # Emit read event
            message = MemoryMessage(
                id=str(uuid.uuid4()),
                namespace=namespace,
                agent_id=agent_id,
                operation="read",
                memory_id=memory_id,
            )
            await self._emit_message(message)
        
        return entry

    async def delete(
        self,
        agent_id: str,
        namespace: str,
        memory_id: str,
    ) -> bool:
        """
        Delete memory from the bus with permission check.
        
        Args:
            agent_id: Deleting agent
            namespace: Source namespace
            memory_id: Memory identifier
            
        Returns:
            True if deleted, False if not found or permission denied
        """
        # Check permission
        target_path = f"{namespace}/{memory_id}"
        perm_ctx = await self._permission_engine.check(agent_id, "delete", target_path)
        
        if not perm_ctx.granted:
            return False
        
        async with self._lock:
            if namespace not in self._memories:
                return False
            if memory_id not in self._memories[namespace]:
                return False
            
            del self._memories[namespace][memory_id]
        
        # Emit delete event
        message = MemoryMessage(
            id=str(uuid.uuid4()),
            namespace=namespace,
            agent_id=agent_id,
            operation="delete",
            memory_id=memory_id,
        )
        await self._emit_message(message)
        await self._persist()
        
        return True

    async def list_memories(
        self,
        agent_id: str,
        namespace: str,
        limit: int = 100,
    ) -> list[SharedMemoryEntry]:
        """List all memories in a namespace that the agent can access."""
        # Check if agent has any access to this namespace
        target_path = f"{namespace}/"
        perm_ctx = await self._permission_engine.check(agent_id, "read", target_path)
        
        if not perm_ctx.granted:
            return []
        
        async with self._lock:
            if namespace not in self._memories:
                return []
            entries = list(self._memories[namespace].values())[:limit]
        
        return entries

    # =========================================================================
    # Subscription System
    # =========================================================================

    async def subscribe(
        self,
        agent_id: str,
        namespace: str,
        filter_pattern: str | None = None,
        event_types: list[str] | None = None,
    ) -> str:
        """
        Subscribe an agent to memory change events.
        
        Args:
            agent_id: Agent to subscribe
            namespace: Namespace to subscribe to
            filter_pattern: Optional glob pattern to filter memory IDs
            event_types: List of event types to subscribe to (read/write/delete)
            
        Returns:
            Subscription ID
        """
        if namespace not in self._namespaces:
            await self.create_namespace(namespace)
        
        subscription = Subscription(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            namespace=namespace,
            filter_pattern=filter_pattern,
            event_types=event_types or ["read", "write", "delete"],
        )
        
        async with self._lock:
            if namespace not in self._subscriptions:
                self._subscriptions[namespace] = []
            self._subscriptions[namespace].append(subscription)
        
        return subscription.id

    async def unsubscribe(self, agent_id: str, subscription_id: str) -> bool:
        """Unsubscribe an agent from a subscription."""
        async with self._lock:
            for namespace, subs in self._subscriptions.items():
                for i, sub in enumerate(subs):
                    if sub.id == subscription_id and sub.agent_id == agent_id:
                        subs.pop(i)
                        return True
        return False

    async def get_subscriptions(self, agent_id: str) -> list[Subscription]:
        """Get all subscriptions for an agent."""
        result = []
        async with self._lock:
            for subs in self._subscriptions.values():
                result.extend([s for s in subs if s.agent_id == agent_id])
        return result

    # =========================================================================
    # Message Emission
    # =========================================================================

    async def _emit_message(self, message: MemoryMessage) -> None:
        """Emit a memory message to all relevant subscribers."""
        async with self._lock:
            subscriptions = list(self._subscriptions.get(message.namespace, []))
        
        for sub in subscriptions:
            if sub.agent_id == message.agent_id:
                continue  # Don't notify the agent that triggered the event
            
            if message.operation not in sub.event_types:
                continue
            
            if sub.filter_pattern:
                import fnmatch
                if not fnmatch.fnmatch(message.memory_id, sub.filter_pattern):
                    continue
            
            # Deliver message to subscriber
            await self._deliver_message(sub.agent_id, message)

    async def _deliver_message(self, agent_id: str, message: MemoryMessage) -> None:
        """Deliver a message to a specific agent."""
        handlers = self._event_handlers.get(agent_id, [])
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Message delivery failed: {e}")

    def add_event_handler(
        self,
        agent_id: str,
        handler: Callable[[MemoryMessage], Awaitable[None]],
    ) -> None:
        """Add an async event handler for an agent."""
        if agent_id not in self._event_handlers:
            self._event_handlers[agent_id] = []
        self._event_handlers[agent_id].append(handler)

    def remove_event_handler(
        self,
        agent_id: str,
        handler: Callable[[MemoryMessage], Awaitable[None]],
    ) -> None:
        """Remove an event handler for an agent."""
        if agent_id in self._event_handlers:
            if handler in self._event_handlers[agent_id]:
                self._event_handlers[agent_id].remove(handler)

    # =========================================================================
    # Persistence
    # =========================================================================

    async def _persist(self) -> None:
        """Persist memory bus state to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "namespaces": list(self._namespaces),
            "memories": {},
        }
        
        async with self._lock:
            for ns, memories in self._memories.items():
                data["memories"][ns] = {
                    mid: asdict(entry) for mid, entry in memories.items()
                }
        
        tmp = self._bus_file.with_suffix(".tmp")
        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(self._bus_file)

    async def load(self) -> None:
        """Load memory bus state from disk."""
        if not self._bus_file.exists():
            return
        
        async with aiofiles.open(self._bus_file, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
        
        async with self._lock:
            self._namespaces = set(data.get("namespaces", ["default"]))
            self._memories = {}
            
            for ns, memories_data in data.get("memories", {}).items():
                self._memories[ns] = {
                    mid: SharedMemoryEntry(**entry_data)
                    for mid, entry_data in memories_data.items()
                }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "MemoryBus",
    "MemoryMessage",
    "SharedMemoryEntry",
    "Subscription",
]

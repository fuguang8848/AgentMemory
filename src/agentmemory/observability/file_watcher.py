"""
File Watcher — Monitor Source Code Changes and Trigger Incremental Reindex

SpectrAI Enhancement: v0.4 §6.3
- Monitor source directory for changes
- Automatically trigger incremental reindex
- Record change events to memory
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable
from dataclasses import dataclass, field

try:
    import watchdog.observers
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class FileChangeEvent:
    """File change event"""
    event_type: str  # created, modified, deleted
    file_path: str
    timestamp: str = ""
    change_type: str = ""  # source, config, test
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass 
class ReindexTask:
    """Reindex task for incremental processing"""
    file_path: str
    operation: str  # add, update, delete
    priority: int = 0


# ============================================================================
# File Change Handler
# ============================================================================


class AgentMemoryFileHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """
    File system event handler for AgentMemory source changes.
    
    Detects changes to source files and queues reindex tasks.
    """

    def __init__(
        self,
        source_dir: Path,
        reindex_callback: Callable[[ReindexTask], Awaitable[None]] | None = None,
        memory_record_callback: Callable[[FileChangeEvent], Awaitable[None]] | None = None,
        ignore_patterns: list[str] | None = None,
    ):
        """
        Initialize file handler.
        
        Args:
            source_dir: Directory to monitor
            reindex_callback: Async callback when file changes (for reindex)
            memory_record_callback: Async callback to record events to memory
            ignore_patterns: Glob patterns to ignore
        """
        self.source_dir = Path(source_dir)
        self.reindex_callback = reindex_callback
        self.memory_record_callback = memory_record_callback
        self.ignore_patterns = ignore_patterns or [
            "*.pyc", "__pycache__", ".git", ".pytest_cache",
            "*.egg-info", ".tox", ".mypy_cache", ".ruff_cache",
            "node_modules", ".venv", "venv",
        ]
        
        self._event_queue: asyncio.Queue[FileChangeEvent] = asyncio.Queue()
        self._reindex_queue: asyncio.Queue[ReindexTask] = asyncio.Queue()
        self._processed_paths: set[str] = set()
        self._logger = logging.getLogger(__name__)

    def _should_ignore(self, path: str) -> bool:
        """Check if path should be ignored."""
        import fnmatch
        path_obj = Path(path)
        
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
            if fnmatch.fnmatch(path_obj.name, pattern):
                return True
        return False

    def _classify_change(self, path: str) -> str:
        """Classify the type of file change."""
        path_obj = Path(path)
        name = path_obj.name
        
        if name.startswith("test_") or name.endswith("_test.py"):
            return "test"
        if name in ("__init__.py", "__main__.py"):
            return "source"
        if path_obj.suffix in (".yaml", ".yml", ".toml", ".json", ".ini", ".cfg"):
            return "config"
        if path_obj.suffix in (".py", ".pyx", ".pxd"):
            return "source"
        return "other"

    def _dispatch(self, event):
        """Handle file system events."""
        if not WATCHDOG_AVAILABLE:
            return
            
        if event.is_directory:
            return
        
        path = event.src_path
        
        if self._should_ignore(path):
            return
        
        # Classify event type
        if isinstance(event, FileCreatedEvent):
            event_type = "created"
        elif isinstance(event, FileModifiedEvent):
            event_type = "modified"
        elif isinstance(event, FileDeletedEvent):
            event_type = "deleted"
        else:
            event_type = "unknown"
        
        change_type = self._classify_change(path)
        
        change_event = FileChangeEvent(
            event_type=event_type,
            file_path=path,
            change_type=change_type,
        )
        
        # Queue event
        self._event_queue.put_nowait(change_event)
        
        # Create reindex task
        if change_type == "source":
            operation = "add" if event_type == "created" else ("update" if event_type == "modified" else "delete")
            reindex_task = ReindexTask(
                file_path=path,
                operation=operation,
                priority=1 if event_type == "created" else 0,
            )
            self._reindex_queue.put_nowait(reindex_task)
            
            # Trigger reindex callback
            if self.reindex_callback:
                asyncio.create_task(self._safe_callback(self.reindex_callback, reindex_task))
        
        # Record to memory
        if self.memory_record_callback:
            asyncio.create_task(self._safe_callback(self.memory_record_callback, change_event))
        
        self._logger.debug(f"File {event_type}: {path} ({change_type})")

    async def _safe_callback(self, callback: Callable, arg) -> None:
        """Safely execute a callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(arg)
            else:
                callback(arg)
        except Exception as e:
            self._logger.error(f"Callback failed: {e}")

    # Watchdog event handlers
    if WATCHDOG_AVAILABLE:
        def on_created(self, event):
            self._dispatch(event)
        
        def on_modified(self, event):
            self._dispatch(event)
        
        def on_deleted(self, event):
            self._dispatch(event)


# ============================================================================
# File Watcher Service
# ============================================================================


class FileWatcher:
    """
    File watcher service for monitoring AgentMemory source changes.
    
    Features:
    - Monitor source directory for file changes
    - Debounce rapid changes
    - Queue and batch reindex tasks
    - Record change events to memory
    
    Usage:
        watcher = FileWatcher(
            source_dir="/path/to/src/agentmemory",
            memory_bus=memory_bus,
        )
        await watcher.start()
        
        # Later...
        await watcher.stop()
    """

    def __init__(
        self,
        source_dir: Path | str,
        reindex_callback: Callable[[ReindexTask], Awaitable[None]] | None = None,
        memory_bus: Any = None,  # Optional MemoryBus for recording events
        ignore_patterns: list[str] | None = None,
        debounce_seconds: float = 1.0,
    ):
        """
        Initialize FileWatcher.
        
        Args:
            source_dir: Directory to monitor
            reindex_callback: Async callback for reindex tasks
            memory_bus: Optional MemoryBus for recording events
            ignore_patterns: Patterns to ignore
            debounce_seconds: Seconds to wait before processing batch
        """
        self.source_dir = Path(source_dir)
        self.reindex_callback = reindex_callback
        self.memory_bus = memory_bus
        self.debounce_seconds = debounce_seconds
        
        self._handler = AgentMemoryFileHandler(
            source_dir=self.source_dir,
            reindex_callback=self._handle_reindex,
            memory_record_callback=self._record_to_memory,
            ignore_patterns=ignore_patterns,
        )
        
        self._observer = None
        self._running = False
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._logger = logging.getLogger(__name__)

    async def _handle_reindex(self, task: ReindexTask) -> None:
        """Internal handler for reindex tasks."""
        if self.reindex_callback:
            await self.reindex_callback(task)

    async def _record_to_memory(self, event: FileChangeEvent) -> None:
        """Record file change event to memory."""
        if not self.memory_bus:
            return
        
        try:
            # Check if memory_bus has write method
            if hasattr(self.memory_bus, 'write'):
                await self.memory_bus.write(
                    agent_id="file_watcher",
                    namespace="system",
                    memory_id=f"file_change:{event.file_path}",
                    content=f"{event.event_type}: {event.file_path}",
                    metadata={
                        "event_type": event.event_type,
                        "change_type": event.change_type,
                        "timestamp": event.timestamp,
                    },
                )
        except Exception as e:
            self._logger.warning(f"Failed to record to memory: {e}")

    async def start(self) -> None:
        """Start watching for file changes."""
        if not WATCHDOG_AVAILABLE:
            self._logger.warning("watchdog not available, file watching disabled")
            return
        
        if self._running:
            return
        
        self._running = True
        self._event_loop = asyncio.get_event_loop()
        
        # Start watchdog observer
        self._observer = watchdog.observers.Observer()
        self._observer.schedule(self._handler, str(self.source_dir), recursive=True)
        self._observer.start()
        
        self._logger.info(f"Started watching: {self.source_dir}")

    async def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._running:
            return
        
        self._running = False
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        
        self._logger.info("Stopped file watching")

    async def get_pending_events(self) -> list[FileChangeEvent]:
        """Get pending file change events."""
        events = []
        while not self._handler._event_queue.empty():
            try:
                events.append(self._handler._event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events

    async def get_pending_reindex_tasks(self) -> list[ReindexTask]:
        """Get pending reindex tasks."""
        tasks = []
        while not self._handler._reindex_queue.empty():
            try:
                tasks.append(self._handler._reindex_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return tasks

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "FileWatcher",
    "FileChangeEvent",
    "ReindexTask",
    "AgentMemoryFileHandler",
]

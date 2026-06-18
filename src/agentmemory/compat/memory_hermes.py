"""MemoryHermes compatibility shim - delegates to the new Memory class.

Per ARCHITECTURE.md §6.1 line 881:
    src/memory_manager.py -> src/agentmemory/core/memory.py
    compat/memory_hermes.py provides MemoryHermes = Memory alias.

This module provides backward-compatible access to the old MemoryHermes API
by aliasing the new Memory class. Any existing 1.x code using MemoryHermes
will transparently use the new Memory implementation.
"""

from __future__ import annotations

from agentmemory.core.memory import Memory

# Public alias as specified in ARCHITECTURE.md §6.1
MemoryHermes = Memory

__all__ = ["MemoryHermes"]

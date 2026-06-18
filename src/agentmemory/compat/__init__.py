"""Compatibility shim layer for AgentMemory 2.0.

Per ARCHITECTURE.md §6.1 (lines 877-889):
    This module provides backward-compatible aliases and wrappers that
    allow 1.x code to continue working with the 2.0 architecture.
"""

from __future__ import annotations

from agentmemory.compat.memory_hermes import MemoryHermes

__all__ = ["MemoryHermes"]

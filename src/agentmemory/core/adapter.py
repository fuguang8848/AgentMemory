"""FrameworkAdapter ABC.

References:
    - ARCHITECTURE.md §5 (framework adapter pattern)
"""

from __future__ import annotations

__all__ = ["FrameworkAdapter"]

from abc import ABC, abstractmethod
from typing import Any


class FrameworkAdapter(ABC):
    """Abstract base class for framework adapters.

    Adapters bind AgentMemory to specific Agent frameworks
    (LangChain, LlamaIndex, AutoGen, CrewAI, etc.).
    """

    framework_name: str

    @abstractmethod
    def bind(self, memory: Any) -> Any:
        """Bind this adapter to a Memory instance.

        Args:
            memory: A Memory or MemoryProvider instance

        Returns:
            A framework-specific memory wrapper
        """
        ...

    @abstractmethod
    def export_tools(self) -> list[dict]:
        """Export AgentMemory operations as tools for the framework.

        Returns:
            List of tool definitions (dict format)
        """
        ...

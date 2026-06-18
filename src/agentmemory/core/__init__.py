"""Core module - 11 Protocol/ABC definitions for AgentMemory 2.0.

References:
    - ARCHITECTURE.md §5.2-5.3 (lines 500-660)
    - ARCHITECTURE.md §6 (directory structure)
"""

from __future__ import annotations

__all__ = [
    # Types
    "MemoryItem",
    "SearchQuery",
    "SearchResult",
    "Episode",
    "MemoryType",
    "MemoryLayer",
    # Protocols & ABCs
    "MemoryProvider",
    "Memory",
    "LLMProvider",
    "LLMResponse",
    "Embedder",
    "VectorStore",
    "GraphStore",
    "GraphNode",
    "GraphEdge",
    "FileStore",
    "Retriever",
    "RetrievalStrategy",
    "Reranker",
    "FactExtractor",
    "FrameworkAdapter",
    # Errors
    "AgentMemoryError",
    "ProviderError",
    "StorageError",
    "ValidationError",
    "NotFoundError",
    "AuthenticationError",
    "ConfigurationError",
]

from .types import (
    Episode,
    MemoryItem,
    MemoryLayer,
    MemoryType,
    SearchQuery,
    SearchResult,
)
from .memory import Memory, MemoryProvider
from .llm import LLMProvider, LLMResponse
from .embedder import Embedder
from .vector import VectorStore
from .graph import GraphNode, GraphEdge, GraphStore
from .file_store import FileStore
from .retriever import Retriever, RetrievalStrategy
from .reranker import Reranker
from .extractor import FactExtractor
from .adapter import FrameworkAdapter
from .errors import (
    AgentMemoryError,
    AuthenticationError,
    ConfigurationError,
    NotFoundError,
    ProviderError,
    StorageError,
    ValidationError,
)

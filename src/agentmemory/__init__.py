"""AgentMemory 2.0 - Multi-Memory Provider System

A multi-provider memory system with layered architecture,
supporting vector search, graph knowledge, and multi-agent episodic memory.

Architecture Reference:
    - ARCHITECTURE.md (3050 lines, ~123KB)
    - /home/fuguang/桌面/ARCHITECTURE.md

Providers (Default Matrix):
    - LLM:        OpenAICompatLLM (via OpenRouter/MiniMax)
    - Embedder:   MiniLMEmbedder
    - Vector:     SQLiteVecStore
    - Graph:      NetworkXGraphStore
    - Storage:    SQLiteStorage
    - Reranker:   IdentityReranker
    - Extractor:  LLMFactExtractor
    - Decay:      HalfLifeDecay

Pipelines:
    - IngestPipeline  : dedupe -> PII redact -> chunk
    - ExtractPipeline  : LLM fact extraction
    - EmbedPipeline    : batch vectorization w/ caching
    - IndexPipeline    : async storage to VectorStore/GraphStore/FileStore
    - RetrievePipeline : hybrid retrieval (vector + BM25 + importance)
    - DecayPipeline   : half-life forgetting

Usage:
    >>> from agentmemory import create_memory
    >>> memory = create_memory()
    >>> await memory.add("hello world")
"""

from __future__ import annotations

# Core types and ABCs
from agentmemory.core.types import (
    MemoryItem,
    SearchQuery,
    SearchResult,
    Episode,
    MemoryType,
    MemoryLayer,
)
from agentmemory.core.memory import Memory, MemoryProvider
from agentmemory.core.llm import LLMProvider, LLMResponse
from agentmemory.core.embedder import Embedder
from agentmemory.core.vector import VectorStore
from agentmemory.core.graph import GraphNode, GraphEdge, GraphStore
from agentmemory.core.file_store import FileStore
from agentmemory.core.retriever import Retriever, RetrievalStrategy
from agentmemory.core.reranker import Reranker
from agentmemory.core.extractor import FactExtractor
from agentmemory.core.errors import (
    AgentMemoryError,
    ProviderError,
    StorageError,
    ValidationError,
    NotFoundError,
    AuthenticationError,
    ConfigurationError,
)

# Default providers
from agentmemory.providers import (
    LLMProvider as DefaultLLM,
    Embedder as DefaultEmbedder,
    VectorStore as DefaultVectorStore,
    GraphStore as DefaultGraphStore,
    Storage as DefaultStorage,
    Reranker as DefaultReranker,
    FactExtractor as DefaultExtractor,
    DecayPolicy as DefaultDecay,
)

# Pipelines
from agentmemory.pipeline import (
    IngestPipeline,
    ExtractPipeline,
    EmbedPipeline,
    IndexPipeline,
    RetrievePipeline,
    DecayPipeline,
)

# Security
from agentmemory.security.pii_redact import PIIRedactor
from agentmemory.security.rate_limit import RateLimiter
from agentmemory.security.circuit_breaker import CircuitBreaker

# Observability
from agentmemory.observability.logging import get_logger
from agentmemory.observability.metrics import MetricsCollector
from agentmemory.observability.events import EventBus

# Config
from agentmemory.config.loader import load_config

# Compatibility
try:
    from agentmemory.compat.migration import migrate_from_v1
except ImportError:
    migrate_from_v1 = None
try:
    from agentmemory.compat.memory_hermes import MemoryHermes as HermesMemoryAdapter
except ImportError:
    HermesMemoryAdapter = None

__version__ = "2.0.0"

__all__ = [
    # Core
    "MemoryItem",
    "SearchQuery",
    "SearchResult",
    "Episode",
    "MemoryType",
    "MemoryLayer",
    "Memory",
    "MemoryProvider",
    "LLMProvider",
    "LLMResponse",
    "Embedder",
    "VectorStore",
    "GraphNode",
    "GraphEdge",
    "GraphStore",
    "FileStore",
    "Retriever",
    "RetrievalStrategy",
    "Reranker",
    "FactExtractor",
    # Errors
    "AgentMemoryError",
    "ProviderError",
    "StorageError",
    "ValidationError",
    "NotFoundError",
    "AuthenticationError",
    "ConfigurationError",
    # Default providers
    "DefaultLLM",
    "DefaultEmbedder",
    "DefaultVectorStore",
    "DefaultGraphStore",
    "DefaultStorage",
    "DefaultReranker",
    "DefaultExtractor",
    "DefaultDecay",
    # Pipelines
    "IngestPipeline",
    "ExtractPipeline",
    "EmbedPipeline",
    "IndexPipeline",
    "RetrievePipeline",
    "DecayPipeline",
    # Security
    "PIIRedactor",
    "RateLimiter",
    "CircuitBreaker",
    # Observability
    "get_logger",
    "MetricsCollector",
    "EventBus",
    # Config
    "load_config",
    # Compat
    "migrate_from_v1",
    "HermesMemoryAdapter",
    # Version
    "__version__",
]

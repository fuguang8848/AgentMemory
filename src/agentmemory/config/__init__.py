# agentmemory/config/__init__.py
"""AgentMemory configuration layer.

Exports:
    - AgentMemoryConfig: Main configuration model (Pydantic v2)
    - Config: Alias for AgentMemoryConfig (backward compatibility)
    - load_config: Multi-source configuration loader
    - get_config: Convenience function to get config
    - All sub-config models: LLMConfig, EmbedderConfig, VectorStoreConfig, etc.
"""
from .schema import (
    AgentMemoryConfig,
    LLMConfig,
    EmbedderConfig,
    VectorStoreConfig,
    GraphStoreConfig,
    FileStoreConfig,
    StorageConfig,
    DecayConfig,
    RetrievalConfig,
    SecurityConfig,
    ObservabilityConfig,
    MiddlewareConfig,
    TenantConfig,
)
from .loader import load_config, get_config

__all__ = [
    # Main config
    "AgentMemoryConfig",
    "load_config",
    "get_config",
    # Sub-configs
    "LLMConfig",
    "EmbedderConfig",
    "VectorStoreConfig",
    "GraphStoreConfig",
    "FileStoreConfig",
    "StorageConfig",
    "DecayConfig",
    "RetrievalConfig",
    "SecurityConfig",
    "ObservabilityConfig",
    "MiddlewareConfig",
    "TenantConfig",
]

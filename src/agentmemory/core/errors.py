"""Unified exception hierarchy for AgentMemory 2.0.

References:
    - ARCHITECTURE.md §5.3 (exception design)
"""

from __future__ import annotations

__all__ = [
    "AgentMemoryError",
    "ProviderError",
    "StorageError",
    "ValidationError",
    "NotFoundError",
    "AuthenticationError",
    "ConfigurationError",
]


class AgentMemoryError(Exception):
    """Base exception for all AgentMemory errors."""

    pass


class ProviderError(AgentMemoryError):
    """Raised when a provider (LLM/Embedder) fails."""

    pass


class StorageError(AgentMemoryError):
    """Raised when a storage backend (vector/graph/file) fails."""

    pass


class ValidationError(AgentMemoryError):
    """Raised when input validation fails."""

    pass


class NotFoundError(AgentMemoryError):
    """Raised when a requested resource is not found."""

    pass


class AuthenticationError(AgentMemoryError):
    """Raised when authentication/authorization fails."""

    pass


class ConfigurationError(AgentMemoryError):
    """Raised when configuration is invalid or missing."""

    pass

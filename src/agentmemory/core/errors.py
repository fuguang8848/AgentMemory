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

    def __str__(self):
        return f"AgentMemoryError: {super().__str__()}"


class ProviderError(AgentMemoryError):
    """Raised when a provider (LLM/Embedder) fails."""

    pass

    def __init__(self, message: str = "", provider: str = ""):
        super().__init__(message)
        self.provider = provider

    def __str__(self):
        return f"ProviderError({self.provider}): {super().__str__()}"


class StorageError(AgentMemoryError):
    """Raised when a storage backend (vector/graph/file) fails."""

    pass

    def __init__(self, message: str = "", backend: str = ""):
        super().__init__(message)
        self.backend = backend

    def __str__(self):
        return f"StorageError({self.backend}): {super().__str__()}"


class ValidationError(AgentMemoryError):
    """Raised when input validation fails."""

    pass

    def __init__(self, message: str = "", field: str = ""):
        super().__init__(message)
        self.field = field

    def __str__(self):
        return f"ValidationError({self.field}): {super().__str__()}"


class NotFoundError(AgentMemoryError):
    """Raised when a requested resource is not found."""

    pass

    def __init__(self, message: str = "", resource_type: str = "", resource_id: str = ""):
        super().__init__(message)
        self.resource_type = resource_type
        self.resource_id = resource_id

    def __str__(self):
        return f"NotFoundError({self.resource_type}/{self.resource_id}): {super().__str__()}"


class AuthenticationError(AgentMemoryError):
    """Raised when authentication/authorization fails."""

    pass

    def __str__(self):
        return f"AuthenticationError: {super().__str__()}"


class ConfigurationError(AgentMemoryError):
    """Raised when configuration is invalid or missing."""

    pass

    def __init__(self, message: str = "", config_key: str = ""):
        super().__init__(message)
        self.config_key = config_key

    def __str__(self):
        return f"ConfigurationError({self.config_key}): {super().__str__()}"

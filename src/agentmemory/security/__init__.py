"""Security module - PII redaction, rate limiting, circuit breakers.

References:
    - ARCHITECTURE.md §7 (security)
"""

from __future__ import annotations

__all__ = [
    "PIIRedactor",
    "PIIRule",
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitAlgorithm",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "CircuitBreakerRegistry",
]

from .pii_redact import PIIRedactor, PIIRule, DEFAULT_PII_RULES
from .rate_limit import RateLimiter, RateLimitConfig, RateLimitAlgorithm
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    CircuitBreakerRegistry,
)

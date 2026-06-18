"""Rate Limiter - Token bucket and sliding window rate limiting.

References:
    - ARCHITECTURE.md §7.2 (defense in depth)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..observability.events import Event, EventBus, EventType


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithm types."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule."""
    requests: int  # Max requests allowed
    window_seconds: float  # Time window in seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET


@dataclass
class TokenBucketState:
    """Token bucket algorithm state."""
    tokens: float
    last_update: float
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class SlidingWindowState:
    """Sliding window algorithm state."""
    timestamps: deque[float] = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)


class RateLimiter:
    """Rate limiter with token bucket and sliding window algorithms.

    Supports per-key rate limiting with event bus integration.

    Example:
        >>> limiter = RateLimiter(default_limit=RateLimitConfig(requests=100, window_seconds=60.0))
        >>> limiter.is_allowed("user:123")
        True
        >>> limiter.acquire("user:123")
        True
    """

    def __init__(
        self,
        default_limit: RateLimitConfig | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize RateLimiter.

        Args:
            default_limit: Default rate limit configuration.
            event_bus: Optional EventBus for logging rate limit events.
        """
        self._default_limit = default_limit or RateLimitConfig(
            requests=100, window_seconds=60.0
        )
        self._event_bus = event_bus

        # Per-key configurations (key -> config)
        self._configs: dict[str, RateLimitConfig] = {}

        # Per-key state
        self._token_buckets: dict[str, TokenBucketState] = {}
        self._sliding_windows: dict[str, SlidingWindowState] = {}

        self._lock = threading.RLock()
        self._stats = {"total_checks": 0, "allowed": 0, "rejected": 0}

    def set_limit(self, key: str, config: RateLimitConfig) -> None:
        """Set rate limit for a specific key.

        Args:
            key: Rate limit key (e.g., "user:123", "ip:192.168.1.1")
            config: Rate limit configuration.
        """
        with self._lock:
            self._configs[key] = config
            # Reset state when config changes
            self._token_buckets.pop(key, None)
            self._sliding_windows.pop(key, None)

    def get_limit(self, key: str) -> RateLimitConfig:
        """Get rate limit config for a key.

        Args:
            key: Rate limit key.

        Returns:
            Rate limit configuration for this key.
        """
        return self._configs.get(key, self._default_limit)

    def is_allowed(self, key: str) -> bool:
        """Check if a request is allowed without consuming a token.

        Args:
            key: Rate limit key.

        Returns:
            True if request would be allowed under the limit.
        """
        config = self.get_limit(key)

        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._check_token_bucket(key, config)
        else:
            return self._check_sliding_window(key, config)

    def acquire(self, key: str) -> bool:
        """Acquire a token (consume a request slot).

        Args:
            key: Rate limit key.

        Returns:
            True if token acquired (request allowed).
            False if rate limit exceeded.
        """
        config = self.get_limit(key)
        allowed = False

        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            allowed = self._acquire_token_bucket(key, config)
        else:
            allowed = self._acquire_sliding_window(key, config)

        # Update stats
        with self._lock:
            self._stats["total_checks"] += 1
            if allowed:
                self._stats["allowed"] += 1
            else:
                self._stats["rejected"] += 1

        # Emit event if rate limited
        if not allowed and self._event_bus is not None:
            event = Event(
                type=EventType.RATE_LIMITED,
                payload={
                    "key": key,
                    "limit": config.requests,
                    "window": config.window_seconds,
                    "algorithm": config.algorithm.value,
                }
            )
            self._emit_event(event)

        return allowed

    def _check_token_bucket(self, key: str, config: RateLimitConfig) -> bool:
        """Check token bucket without consuming."""
        with self._lock:
            state = self._token_buckets.get(key)
            if state is None:
                return True  # New key, allowed

            with state.lock:
                now = time.time()
                # Refill tokens based on elapsed time
                elapsed = now - state.last_update
                refill = elapsed * (config.requests / config.window_seconds)
                state.tokens = min(config.requests, state.tokens + refill)
                state.last_update = now

                return state.tokens >= 1.0

    def _acquire_token_bucket(self, key: str, config: RateLimitConfig) -> bool:
        """Acquire from token bucket."""
        with self._lock:
            state = self._token_buckets.get(key)
            if state is None:
                # Initialize new bucket
                state = TokenBucketState(
                    tokens=config.requests - 1,  # Consume one
                    last_update=time.time()
                )
                self._token_buckets[key] = state
                return True

            with state.lock:
                now = time.time()
                elapsed = now - state.last_update
                refill = elapsed * (config.requests / config.window_seconds)
                state.tokens = min(config.requests, state.tokens + refill)
                state.last_update = now

                if state.tokens >= 1.0:
                    state.tokens -= 1.0
                    return True
                return False

    def _check_sliding_window(self, key: str, config: RateLimitConfig) -> bool:
        """Check sliding window without consuming."""
        with self._lock:
            state = self._sliding_windows.get(key)
            if state is None:
                return True  # New key, allowed

            with state.lock:
                now = time.time()
                window_start = now - config.window_seconds

                # Remove old timestamps
                while state.timestamps and state.timestamps[0] < window_start:
                    state.timestamps.popleft()

                return len(state.timestamps) < config.requests

    def _acquire_sliding_window(self, key: str, config: RateLimitConfig) -> bool:
        """Acquire from sliding window."""
        with self._lock:
            state = self._sliding_windows.get(key)
            if state is None:
                state = SlidingWindowState()
                self._sliding_windows[key] = state

            with state.lock:
                now = time.time()
                window_start = now - config.window_seconds

                # Remove old timestamps
                while state.timestamps and state.timestamps[0] < window_start:
                    state.timestamps.popleft()

                if len(state.timestamps) < config.requests:
                    state.timestamps.append(now)
                    return True
                return False

    def _emit_event(self, event: Event) -> None:
        """Emit event to event bus."""
        if self._event_bus is None:
            return
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._event_bus.emit(event))
            else:
                loop.run_until_complete(self._event_bus.emit(event))
        except Exception:
            pass  # Don't let event failures affect rate limiting

    def reset(self, key: str | None = None) -> None:
        """Reset rate limit state.

        Args:
            key: Specific key to reset, or None to reset all.
        """
        with self._lock:
            if key is None:
                self._token_buckets.clear()
                self._sliding_windows.clear()
            else:
                self._token_buckets.pop(key, None)
                self._sliding_windows.pop(key, None)

    def stats(self) -> dict[str, Any]:
        """Get rate limiter statistics.

        Returns:
            Dict with total checks, allowed, and rejected counts.
        """
        with self._lock:
            return {
                "total_checks": self._stats["total_checks"],
                "allowed": self._stats["allowed"],
                "rejected": self._stats["rejected"],
            }

    def remaining(self, key: str) -> int:
        """Get remaining requests for a key.

        Args:
            key: Rate limit key.

        Returns:
            Number of remaining requests in current window.
        """
        config = self.get_limit(key)

        if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            with self._lock:
                state = self._token_buckets.get(key)
                if state is None:
                    return config.requests
                with state.lock:
                    now = time.time()
                    elapsed = now - state.last_update
                    refill = elapsed * (config.requests / config.window_seconds)
                    tokens = min(config.requests, state.tokens + refill)
                    return int(tokens)
        else:
            with self._lock:
                state = self._sliding_windows.get(key)
                if state is None:
                    return config.requests
                with state.lock:
                    now = time.time()
                    window_start = now - config.window_seconds
                    while state.timestamps and state.timestamps[0] < window_start:
                        state.timestamps.popleft()
                    return max(0, config.requests - len(state.timestamps))

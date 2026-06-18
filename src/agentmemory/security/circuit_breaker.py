"""Circuit Breaker - Fault tolerance pattern implementation.

References:
    - ARCHITECTURE.md §7.2 (defense in depth)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar, Generic

from ..observability.events import Event, EventBus, EventType


T = TypeVar('T')


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject all
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, circuit_name: str, remaining_timeout: float):
        self.circuit_name = circuit_name
        self.remaining_timeout = remaining_timeout
        super().__init__(f"Circuit '{circuit_name}' is open. Retry in {remaining_timeout:.1f}s")


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 2         # Successes in half-open to close
    timeout_seconds: float = 30.0      # Time before trying half-open
    excluded_exceptions: tuple[type, ...] = ()  # Exceptions that don't count


@dataclass
class CircuitBreakerStats:
    """Runtime statistics for a circuit."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0


class CircuitBreaker:
    """Circuit breaker implementation with three states.

    CLOSED -> OPEN: When failure count exceeds threshold
    OPEN -> HALF_OPEN: After timeout expires
    HALF_OPEN -> CLOSED: When success count exceeds threshold
    HALF_OPEN -> OPEN: On any failure

    Example:
        >>> breaker = CircuitBreaker(name="external_api")
        >>> with breaker:
        ...     result = call_external_api()
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        event_bus: EventBus | None = None,
    ):
        """Initialize CircuitBreaker.

        Args:
            name: Unique identifier for this circuit.
            config: Circuit breaker configuration.
            event_bus: Optional EventBus for logging state changes.
        """
        self.name = name
        self._config = config or CircuitBreakerConfig()
        self._event_bus = event_bus

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._lock = threading.RLock()
        self._stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self._stats

    def is_callable(self) -> bool:
        """Check if a call is allowed in current state."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self._config.timeout_seconds:
                        self._transition_to(CircuitState.HALF_OPEN)
                        return True
                return False
            else:  # HALF_OPEN
                return True

    def remaining_timeout(self) -> float:
        """Get seconds until circuit might close."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0
            if self._last_failure_time is None:
                return 0.0
            elapsed = time.time() - self._last_failure_time
            return max(0.0, self._config.timeout_seconds - elapsed)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1

        if new_state == CircuitState.HALF_OPEN:
            self._success_count = 0

        # Emit event
        if self._event_bus is not None:
            event = Event(
                type=EventType.CIRCUIT_OPENED if new_state == CircuitState.OPEN else EventType.CIRCUIT_CLOSED,
                payload={
                    "circuit": self.name,
                    "from_state": old_state.value,
                    "to_state": new_state.value,
                }
            )
            self._emit_event(event)

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
            pass

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._failure_count = 0
            self._stats.successful_calls += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._stats.failed_calls += 1

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: Function to call.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of func call.

        Raises:
            CircuitBreakerError: If circuit is open.
        """
        with self._lock:
            if not self.is_callable():
                self._stats.rejected_calls += 1
                raise CircuitBreakerError(self.name, self.remaining_timeout())

            self._stats.total_calls += 1

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self._config.excluded_exceptions:
            # Don't count excluded exceptions
            raise
        except Exception:
            self._record_failure()
            raise

    def __enter__(self) -> CircuitBreaker:
        """Context manager entry - check if call allowed."""
        if not self.is_callable():
            self._stats.rejected_calls += 1
            raise CircuitBreakerError(self.name, self.remaining_timeout())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - record success/failure."""
        if exc_type is not None and exc_type not in self._config.excluded_exceptions:
            self._record_failure()
            return False  # Re-raise the exception
        else:
            self._record_success()
            return True

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._stats = CircuitBreakerStats()

    def record_rejected(self) -> None:
        """Record a rejected call without executing."""
        with self._lock:
            self._stats.rejected_calls += 1


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.

    Provides centralized access to named circuit breakers.
    """

    def __init__(self, event_bus: EventBus | None = None):
        """Initialize registry.

        Args:
            event_bus: Optional EventBus for all managed breakers.
        """
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()
        self._event_bus = event_bus
        self._lock = threading.Lock()

    def get(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name.

        Args:
            name: Circuit breaker name.
            config: Optional config (uses default if not provided).

        Returns:
            CircuitBreaker instance.
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    config=config or self._default_config,
                    event_bus=self._event_bus,
                )
            return self._breakers[name]

    def list_circuits(self) -> list[str]:
        """List all registered circuit names."""
        with self._lock:
            return list(self._breakers.keys())

    def stats_summary(self) -> dict[str, CircuitBreakerStats]:
        """Get stats for all circuits."""
        with self._lock:
            return {name: cb.stats.copy() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for cb in self._breakers.values():
                cb.reset()

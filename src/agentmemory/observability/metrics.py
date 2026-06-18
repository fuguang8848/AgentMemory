"""Prometheus metrics collection for AgentMemory 2.0.

References:
    - ARCHITECTURE.md §9.2.2 (MetricsMiddleware)
"""

from __future__ import annotations

import time
import contextlib
from typing import Any, Callable

try:
    from prometheus_client import Counter, Histogram, Gauge, Info
    from prometheus_client.openmetrics.exposition import generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# Default buckets for latency histograms (in seconds)
DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class MetricsCollector:
    """Prometheus metrics collector for AgentMemory.

    Provides counters, histograms, and gauges for observability.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global metrics instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, prefix: str = "agentmemory"):
        """Initialize metrics collector.

        Args:
            prefix: Metric name prefix
        """
        if self._initialized:
            return

        self.prefix = prefix
        self._metrics: dict[str, Any] = {}

        if PROMETHEUS_AVAILABLE:
            self._setup_defaults()

        self._initialized = True

    def _setup_defaults(self) -> None:
        """Setup default metrics."""
        # Latency histograms
        self._histogram(
            "store_latency_seconds",
            "Latency of store operations",
            ("operation", "store_type"),
            buckets=DEFAULT_BUCKETS,
        )
        self._histogram(
            "search_latency_seconds",
            "Latency of search operations",
            ("strategy",),
            buckets=DEFAULT_BUCKETS,
        )
        self._histogram(
            "embed_latency_seconds",
            "Latency of embedding operations",
            ("embedder", "batch_size"),
            buckets=DEFAULT_BUCKETS,
        )
        self._histogram(
            "llm_latency_seconds",
            "Latency of LLM calls",
            ("model", "operation"),
            buckets=DEFAULT_BUCKETS,
        )

        # Counters
        self._counter("memory_items_total", "Total memory items", ("operation", "layer"))
        self._counter("memory_items_stored_total", "Total memory items stored", ("layer",))
        self._counter("memory_items_deleted_total", "Total memory items deleted", ("layer",))
        self._counter("llm_calls_total", "Total LLM calls", ("model", "status"))
        self._counter("embed_calls_total", "Total embedding calls", ("embedder", "status"))
        self._counter("search_requests_total", "Total search requests", ("strategy",))
        self._counter("cache_hits_total", "Total cache hits", ("cache_type",))
        self._counter("cache_misses_total", "Total cache misses", ("cache_type",))

        # Gauges
        self._gauge("queue_size", "Current queue size", ("queue_name",))
        self._gauge("active_items", "Number of active memory items", ("layer",))
        self._gauge("provider_status", "Provider connection status (1=up, 0=down)", ("provider",))

    def _counter(self, name: str, description: str, labels: tuple[str, ...]) -> Counter | None:
        """Create or get a counter metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        full_name = f"{self.prefix}_{name}"
        if full_name not in self._metrics:
            self._metrics[full_name] = Counter(full_name, description, labels)
        return self._metrics[full_name]

    def _histogram(
        self, name: str, description: str, labels: tuple[str, ...], buckets: tuple = DEFAULT_BUCKETS
    ) -> Histogram | None:
        """Create or get a histogram metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        full_name = f"{self.prefix}_{name}"
        if full_name not in self._metrics:
            self._metrics[full_name] = Histogram(full_name, description, labels, buckets=buckets)
        return self._metrics[full_name]

    def _gauge(self, name: str, description: str, labels: tuple[str, ...]) -> Gauge | None:
        """Create or get a gauge metric."""
        if not PROMETHEUS_AVAILABLE:
            return None

        full_name = f"{self.prefix}_{name}"
        if full_name not in self._metrics:
            self._metrics[full_name] = Gauge(full_name, description, labels)
        return self._metrics[full_name]

    def record_latency(
        self, metric_name: str, labels: dict[str, str], duration: float
    ) -> None:
        """Record a latency observation.

        Args:
            metric_name: Metric name (without prefix)
            labels: Label values dict
            duration: Duration in seconds
        """
        if not PROMETHEUS_AVAILABLE:
            return

        full_name = f"{self.prefix}_{metric_name}"
        if full_name in self._metrics:
            metric = self._metrics[full_name]
            metric.labels(**labels).observe(duration)

    def increment(self, metric_name: str, labels: dict[str, str], value: float = 1) -> None:
        """Increment a counter.

        Args:
            metric_name: Metric name (without prefix)
            labels: Label values dict
            value: Increment value (default 1)
        """
        if not PROMETHEUS_AVAILABLE:
            return

        full_name = f"{self.prefix}_{metric_name}"
        if full_name in self._metrics:
            self._metrics[full_name].labels(**labels).inc(value)

    def set_gauge(self, metric_name: str, labels: dict[str, str], value: float) -> None:
        """Set a gauge value.

        Args:
            metric_name: Metric name (without prefix)
            labels: Label values dict
            value: Gauge value
        """
        if not PROMETHEUS_AVAILABLE:
            return

        full_name = f"{self.prefix}_{metric_name}"
        if full_name in self._metrics:
            self._metrics[full_name].labels(**labels).set(value)

    def timed(self, metric_name: str, labels: dict[str, str]) -> Callable:
        """Decorator for timing operations.

        Usage:
            @metrics.timed("store_latency_seconds", {"operation": "upsert", "store_type": "vector"})
            async def my_operation():
                ...

        Args:
            metric_name: Metric name (without prefix)
            labels: Label values dict
        """
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    self.record_latency(metric_name, labels, duration)
            return wrapper
        return decorator

    @contextlib.contextmanager
    def timer(self, metric_name: str, labels: dict[str, str]):
        """Context manager for timing operations.

        Usage:
            async with metrics.timer("search_latency_seconds", {"strategy": "vector"}):
                await do_search()

        Args:
            metric_name: Metric name (without prefix)
            labels: Label values dict
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record_latency(metric_name, labels, duration)

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus text format.

        Returns:
            Metrics as Prometheus text format string
        """
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus not available"
        return generate_latest().decode("utf-8")

    def reset(self) -> None:
        """Reset all metrics (mainly for testing)."""
        for metric in self._metrics.values():
            if hasattr(metric, "clear"):
                metric.clear()

"""
OpenTelemetry Telemetry Layer for AgentMemory 2.0

SpectrAI Enhancement: v0.4 §9.2.2
- Distributed tracing (trace) integration
- P99/P95 latency metrics
- Event export to OTLP endpoint
"""

from __future__ import annotations

import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import BatchMetricReader, OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Span, Status, StatusCode, SpanKind
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    metrics = None
    Status = None
    StatusCode = None


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TelemetryConfig:
    """Telemetry configuration"""
    service_name: str = "agentmemory"
    otlp_endpoint: str | None = None  # e.g., "http://localhost:4317"
    export_interval_ms: int = 5000
    trace_export_timeout_ms: int = 3000
    metric_export_timeout_ms: int = 3000
    enable_tracing: bool = True
    enable_metrics: bool = True
    # Latency percentile buckets (in seconds)
    latency_buckets: tuple = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


# ============================================================================
# OTel Telemetry Core
# ============================================================================


class OTelTelemetry:
    """
    OpenTelemetry telemetry layer for AgentMemory.
    
    Provides:
    - Distributed tracing with span creation
    - P99/P95 latency metrics
    - OTLP export to configured endpoint
    
    Usage:
        telemetry = OTelTelemetry(
            service_name="agentmemory",
            otlp_endpoint="http://localhost:4317"
        )
        
        with telemetry.start_span("search") as span:
            # do work
            pass
        
        # P99/P95 metrics are automatically recorded
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: TelemetryConfig | None = None):
        if self._initialized:
            return
        
        self.config = config or TelemetryConfig()
        self._tracer = None
        self._meter = None
        self._tracer_provider = None
        self._meter_provider = None
        
        # Latency storage for percentile calculation
        self._latency_samples: dict[str, list[float]] = {}
        self._latency_lock = asyncio.Lock()
        
        if OTEL_AVAILABLE:
            self._setup_tracing()
            self._setup_metrics()
        
        self._initialized = True

    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing."""
        if not self.config.enable_tracing:
            return
        
        resource = Resource.create({
            "service.name": self.config.service_name,
            "service.version": "2.0",
        })
        
        self._tracer_provider = TracerProvider(resource=resource)
        
        if self.config.otlp_endpoint:
            try:
                span_exporter = OTLPSpanExporter(
                    endpoint=self.config.otlp_endpoint,
                    insecure=True,
                )
                span_processor = BatchSpanProcessor(span_exporter)
                self._tracer_provider.add_span_processor(span_processor)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to setup OTLP trace export: {e}")
        
        trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer(self.config.service_name)

    def _setup_metrics(self) -> None:
        """Setup OpenTelemetry metrics."""
        if not self.config.enable_metrics:
            return
        
        resource = Resource.create({
            "service.name": self.config.service_name,
        })
        
        if self.config.otlp_endpoint:
            try:
                metric_exporter = OTLPMetricExporter(
                    endpoint=self.config.otlp_endpoint,
                    insecure=True,
                )
                metric_reader = BatchMetricReader(metric_exporter)
                self._meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to setup OTLP metric export: {e}")
                self._meter_provider = MeterProvider(resource=resource)
        else:
            self._meter_provider = MeterProvider(resource=resource)
        
        metrics.set_meter_provider(self._meter_provider)
        self._meter = metrics.get_meter(self.config.service_name)
        
        # Create latency histogram instruments
        self._create_latency_histograms()

    def _create_latency_histograms(self) -> None:
        """Create latency histogram instruments for percentile tracking."""
        if not OTEL_AVAILABLE or self._meter is None:
            return
        
        # Store histograms for manual percentile calculation
        self._histograms: dict[str, Any] = {}
        
        for operation in ["store", "search", "embed", "llm", "index"]:
            histogram = self._meter.create_histogram(
                name=f"{operation}_latency_seconds",
                description=f"Latency of {operation} operations",
                unit="s",
            )
            self._histograms[operation] = histogram

    # =========================================================================
    # Tracing
    # =========================================================================

    @property
    def tracer(self):
        """Get the tracer instance."""
        return self._tracer

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: str = "internal",
    ):
        """
        Start a new span context manager.
        
        Args:
            name: Span name
            attributes: Optional span attributes
            kind: Span kind (internal, server, client, producer, consumer)
            
        Yields:
            Span object or None if OTel not available
        """
        if not OTEL_AVAILABLE or self._tracer is None:
            yield None
            return
        
        span_kind = getattr(SpanKind, kind.upper(), SpanKind.INTERNAL)
        
        with self._tracer.start_as_current_span(name, kind=span_kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def get_current_span(self) -> Span | None:
        """Get the current active span."""
        if not OTEL_AVAILABLE:
            return None
        return trace.get_current_span()

    def get_trace_context(self) -> tuple[str | None, str | None]:
        """Get current trace_id and span_id."""
        if not OTEL_AVAILABLE:
            return None, None
        
        span = self.get_current_span()
        if span is None:
            return None, None
        
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return None, None
        
        trace_id = format(ctx.trace_id, "032x")
        span_id = format(ctx.span_id, "016x")
        return trace_id, span_id

    # =========================================================================
    # Latency Metrics
    # =========================================================================

    def record_latency(
        self,
        operation: str,
        duration: float,
        attributes: dict[str, str] | None = None,
    ) -> None:
        """
        Record a latency observation.
        
        Args:
            operation: Operation type (store, search, embed, llm, index)
            duration: Duration in seconds
            attributes: Optional attributes
        """
        if not OTEL_AVAILABLE:
            return
        
        # Record to histogram
        if operation in self._histograms:
            self._histograms[operation].record(duration, attributes or {})
        
        # Store sample for percentile calculation
        self._store_latency_sample(operation, duration)

    async def _store_latency_sample(self, operation: str, duration: float) -> None:
        """Store a latency sample for percentile calculation."""
        async with self._latency_lock:
            if operation not in self._latency_samples:
                self._latency_samples[operation] = []
            self._latency_samples[operation].append(duration)
            
            # Keep only last 10000 samples
            if len(self._latency_samples[operation]) > 10000:
                self._latency_samples[operation] = self._latency_samples[operation][-10000:]

    def calculate_percentile(self, operation: str, percentile: float) -> float | None:
        """
        Calculate latency percentile.
        
        Args:
            operation: Operation type
            percentile: Percentile (0-100), e.g., 99 for P99
            
        Returns:
            Percentile value in seconds, or None if no data
        """
        if operation not in self._latency_samples:
            return None
        
        samples = sorted(self._latency_samples[operation])
        if not samples:
            return None
        
        index = int(len(samples) * (percentile / 100))
        index = min(index, len(samples) - 1)
        return samples[index]

    def get_latency_stats(self, operation: str) -> dict[str, float | None]:
        """
        Get latency statistics for an operation.
        
        Returns:
            Dict with p50, p95, p99, mean, min, max
        """
        if operation not in self._latency_samples or not self._latency_samples[operation]:
            return {"p50": None, "p95": None, "p99": None, "mean": None, "min": None, "max": None}
        
        samples = sorted(self._latency_samples[operation])
        return {
            "p50": self._percentile(samples, 50),
            "p95": self._percentile(samples, 95),
            "p99": self._percentile(samples, 99),
            "mean": sum(samples) / len(samples),
            "min": samples[0],
            "max": samples[-1],
        }

    def _percentile(self, sorted_samples: list[float], percentile: float) -> float:
        """Calculate percentile from sorted samples."""
        if not sorted_samples:
            return 0.0
        index = int(len(sorted_samples) * (percentile / 100))
        index = min(index, len(sorted_samples) - 1)
        return sorted_samples[index]

    # =========================================================================
    # Timed Context Manager / Decorator
    # =========================================================================

    @asynccontextmanager
    async def timed_span(
        self,
        operation: str,
        name: str,
        attributes: dict[str, Any] | None = None,
    ):
        """
        Combined timing and span context manager.
        
        Automatically records latency and creates a span.
        
        Usage:
            async with telemetry.timed_span("search", "hybrid_search"):
                await do_search()
        """
        start = time.perf_counter()
        
        async with self.start_span(name, attributes) as span:
            try:
                yield span
            finally:
                duration = time.perf_counter() - start
                self.record_latency(operation, duration, attributes)

    def timed(self, operation: str) -> Callable:
        """
        Decorator for timing operations.
        
        Usage:
            @telemetry.timed("search")
            async def my_search():
                ...
        """
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    self.record_latency(operation, duration)
            return wrapper
        return decorator

    # =========================================================================
    # Event Export
    # =========================================================================

    async def export_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        trace_context: dict[str, str] | None = None,
    ) -> None:
        """
        Export an event to OTLP endpoint.
        
        Args:
            event_type: Event type name
            payload: Event payload
            trace_context: Optional trace context for linking
        """
        if not OTEL_AVAILABLE:
            return
        
        # Create a span for the event export
        async with self.start_span(f"event.export.{event_type}") as span:
            if span:
                span.set_attribute("event.type", event_type)
                span.set_attribute("event.payload", str(payload))
            
            if trace_context:
                trace_id = trace_context.get("trace_id")
                span_id = trace_context.get("span_id")
                if trace_id:
                    span.set_attribute("event.trace_id", trace_id)
                if span_id:
                    span.set_attribute("event.span_id", span_id)
            
            # Event is exported as part of the span
            # OTLP handles the export

    # =========================================================================
    # Shutdown
    # =========================================================================

    async def shutdown(self) -> None:
        """Shutdown telemetry providers gracefully."""
        if self._tracer_provider:
            if hasattr(self._tracer_provider, 'shutdown'):
                await self._tracer_provider.shutdown()
        
        if self._meter_provider:
            if hasattr(self._meter_provider, 'shutdown'):
                await self._meter_provider.shutdown()


# ============================================================================
# Convenience Functions
# ============================================================================

_telemetry_instance: OTelTelemetry | None = None


def get_telemetry(config: TelemetryConfig | None = None) -> OTelTelemetry:
    """Get or create the global telemetry instance."""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = OTelTelemetry(config)
    return _telemetry_instance


@asynccontextmanager
async def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: str = "internal",
):
    """Convenience function for creating spans."""
    telemetry = get_telemetry()
    async with telemetry.start_span(name, attributes, kind) as span:
        yield span


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "OTelTelemetry",
    "TelemetryConfig",
    "get_telemetry",
    "trace_span",
    "AgentEvaluator",
    "OTelExporter",
]


# ============================================================================
# AgentEvaluator — RagaAI-compatible agent evaluation
# ============================================================================


@dataclass
class EvaluationResult:
    """Result of an agent evaluation."""
    metric: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentProfile:
    """Agent execution profile."""
    agent_name: str
    total_calls: int = 0
    total_duration: float = 0.0
    memory_usage_samples: list[float] = field(default_factory=list)
    error_count: int = 0
    evaluation_results: list[EvaluationResult] = field(default_factory=list)

    @property
    def avg_duration(self) -> float:
        return self.total_duration / self.total_calls if self.total_calls else 0.0

    @property
    def avg_memory_mb(self) -> float:
        return sum(self.memory_usage_samples) / len(self.memory_usage_samples) if self.memory_usage_samples else 0.0


class AgentEvaluator:
    """
    RagaAI-compatible agent evaluator.

    Provides:
    - evaluate_response(): score agent responses
    - track_memory_usage(): track memory consumption
    - profile_agent(): aggregate execution profiling

    Usage:
        evaluator = AgentEvaluator()
        result = await evaluator.evaluate_response(
            agent_name="search_agent",
            prompt="Find documents about AI",
            response="Found 5 documents...",
            expected_outcome="Returns relevant documents",
        )
        print(result.score, result.metric)
    """

    def __init__(self, telemetry: OTelTelemetry | None = None):
        self._telemetry = telemetry
        self._profiles: dict[str, AgentProfile] = {}
        self._memory_lock = asyncio.Lock()

    # --------------------------------------------------------------------------
    # evaluate_response
    # --------------------------------------------------------------------------

    async def evaluate_response(
        self,
        agent_name: str,
        prompt: str,
        response: str,
        expected_outcome: str | None = None,
        metrics: list[str] | None = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate an agent response against multiple metrics.

        Args:
            agent_name: Name of the agent
            prompt: Input prompt
            response: Agent's response
            expected_outcome: Optional expected outcome for comparison
            metrics: List of metrics to evaluate (defaults to ["relevance", "coherence", "correctness"])

        Returns:
            List of EvaluationResult objects
        """
        results: list[EvaluationResult] = []
        default_metrics = ["relevance", "coherence", "correctness"]
        eval_metrics = metrics or default_metrics

        for metric in eval_metrics:
            score = await self._compute_metric(metric, prompt, response, expected_outcome)
            result = EvaluationResult(
                metric=metric,
                score=score,
                details={
                    "agent_name": agent_name,
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                },
            )
            results.append(result)

            # Update profile
            await self._update_profile(agent_name, "evaluation", {"metric": metric, "score": score})

        return results

    async def _update_profile(
        self, agent_name: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """Update agent profile with an event."""
        async with self._memory_lock:
            if agent_name not in self._profiles:
                self._profiles[agent_name] = AgentProfile(agent_name=agent_name)
            # Evaluation results are tracked separately; no-op here
            _ = event_type, data

    async def _compute_metric(
        self,
        metric: str,
        prompt: str,
        response: str,
        expected_outcome: str | None,
    ) -> float:
        """Compute a single metric score (placeholder — plug in LLM judge or heuristics)."""
        # Simple heuristic fallback when no LLM judge is configured
        # Returns a dummy score for demonstration; replace with real evaluation logic
        if metric == "relevance":
            # Rough heuristic: check response length relative to prompt
            ratio = len(response) / max(len(prompt), 1)
            return min(1.0, ratio / 2)
        elif metric == "coherence":
            # Placeholder: coherence score
            return 0.75
        elif metric == "correctness":
            # Placeholder: correctness score
            return 0.80
        return 0.5

    # --------------------------------------------------------------------------
    # track_memory_usage
    # --------------------------------------------------------------------------

    async def track_memory_usage(self, agent_name: str, memory_mb: float) -> None:
        """
        Record a memory usage sample for an agent.

        Args:
            agent_name: Name of the agent
            memory_mb: Memory usage in megabytes
        """
        async with self._memory_lock:
            if agent_name not in self._profiles:
                self._profiles[agent_name] = AgentProfile(agent_name=agent_name)
            self._profiles[agent_name].memory_usage_samples.append(memory_mb)

            # Emit to OTel metrics if available
            if OTEL_AVAILABLE:
                self._record_memory_metric(agent_name, memory_mb)

    def _record_memory_metric(self, agent_name: str, memory_mb: float) -> None:
        """Record memory metric to OTel."""
        telemetry = self._telemetry or get_telemetry()
        if telemetry._meter:
            gauge = telemetry._meter.create_histogram(
                name="agent_memory_mb",
                description="Agent memory usage in MB",
                unit="MB",
            )
            gauge.record(memory_mb, {"agent.name": agent_name})

    # --------------------------------------------------------------------------
    # profile_agent
    # --------------------------------------------------------------------------

    async def profile_agent(
        self,
        agent_name: str,
        duration: float,
        success: bool = True,
        error: str | None = None,
    ) -> AgentProfile:
        """
        Record an agent execution profile.

        Args:
            agent_name: Name of the agent
            duration: Execution duration in seconds
            success: Whether execution succeeded
            error: Optional error message

        Returns:
            Updated AgentProfile
        """
        async with self._memory_lock:
            if agent_name not in self._profiles:
                self._profiles[agent_name] = AgentProfile(agent_name=agent_name)

            profile = self._profiles[agent_name]
            profile.total_calls += 1
            profile.total_duration += duration
            if not success:
                profile.error_count += 1

            # Emit to OTel tracing if available
            if OTEL_AVAILABLE and self._telemetry:
                span_name = f"agent.profile.{agent_name}"
                async with self._telemetry.start_span(span_name) as span:
                    if span:
                        span.set_attribute("agent.name", agent_name)
                        span.set_attribute("agent.duration_s", duration)
                        span.set_attribute("agent.success", success)
                        if error:
                            span.set_attribute("agent.error", error)
                            span.set_status(Status(StatusCode.ERROR, error))

        return self._profiles[agent_name]

    def get_profile(self, agent_name: str) -> AgentProfile | None:
        """Get the profile for an agent."""
        return self._profiles.get(agent_name)

    def get_all_profiles(self) -> dict[str, AgentProfile]:
        """Get all agent profiles."""
        return dict(self._profiles)


# ============================================================================
# OTelExporter — export AgentMemory telemetry to OTLP / console
# ============================================================================


class OTelExporter:
    """
    Export AgentMemory observability data to OTLP endpoints or console.

    Provides:
    - export_to_otel_endpoint(): push traces/metrics to OTLP
    - export_to_console(): print structured telemetry to stdout

    Usage:
        exporter = OTelExporter(
            service_name="agentmemory",
            otlp_endpoint="http://localhost:4317",
        )
        await exporter.export_to_otel_endpoint(telemetry)
        await exporter.export_to_console(evaluation_results)
    """

    def __init__(
        self,
        service_name: str = "agentmemory",
        otlp_endpoint: str | None = None,
    ):
        self._service_name = service_name
        self._otlp_endpoint = otlp_endpoint
        self._telemetry = get_telemetry(
            TelemetryConfig(
                service_name=service_name,
                otlp_endpoint=otlp_endpoint,
            )
        )

    # --------------------------------------------------------------------------
    # export_to_otel_endpoint
    # --------------------------------------------------------------------------

    async def export_to_otel_endpoint(
        self,
        data: dict[str, Any] | list[Any] | None = None,
    ) -> bool:
        """
        Export telemetry data to the configured OTLP endpoint.

        Args:
            data: Optional data payload to export as a span event

        Returns:
            True if export succeeded, False otherwise
        """
        if not OTEL_AVAILABLE:
            import logging
            logging.getLogger(__name__).warning("OTel unavailable — skipping OTLP export")
            return False

        try:
            async with self._telemetry.start_span("otel.export") as span:
                if span and data is not None:
                    span.set_attribute("export.service_name", self._service_name)
                    span.set_attribute("export.otlp_endpoint", self._otlp_endpoint or "none")
                    span.set_attribute("export.payload", str(data)[:500])

                # Flush tracer provider to push any pending spans
                if self._telemetry._tracer_provider:
                    if hasattr(self._telemetry._tracer_provider, "force_flush"):
                        self._telemetry._tracer_provider.force_flush()

                if span:
                    span.set_attribute("export.success", True)
                return True

        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"OTLP export failed: {e}")
            return False

    # --------------------------------------------------------------------------
    # export_to_console
    # --------------------------------------------------------------------------

    async def export_to_console(
        self,
        data: dict[str, Any] | list[EvaluationResult] | TelemetryConfig | None = None,
        title: str | None = None,
    ) -> None:
        """
        Print structured telemetry data to the console.

        Args:
            data: Data to export (EvaluationResult list, dict, or TelemetryConfig)
            title: Optional section title
        """
        ts = datetime.now(timezone.utc).isoformat()

        print(f"\n{'=' * 60}")
        if title:
            print(f"  {title}")
        print(f"  Timestamp: {ts}")
        print(f"  Service: {self._service_name}")
        print('=' * 60)

        if data is None:
            print("  (no data)")
        elif isinstance(data, list) and data and isinstance(data[0], EvaluationResult):
            for r in data:
                print(f"  [{r.metric}] score={r.score:.3f}  details={r.details}")
        elif isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        elif isinstance(data, TelemetryConfig):
            print(f"  service_name: {data.service_name}")
            print(f"  otlp_endpoint: {data.otlp_endpoint}")
            print(f"  enable_tracing: {data.enable_tracing}")
            print(f"  enable_metrics: {data.enable_metrics}")
        else:
            print(f"  {data}")

        print('-' * 60 + '\n')

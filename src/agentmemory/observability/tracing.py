"""OpenTelemetry tracing for AgentMemory 2.0.

References:
    - ARCHITECTURE.md §9.2.2 (TracingMiddleware)
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace import Span, Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


class TracingContext:
    """OpenTelemetry tracing context manager.

    Provides span creation and context propagation for distributed tracing.
    """

    def __init__(self, service_name: str = "agentmemory"):
        """Initialize TracingContext.

        Args:
            service_name: Name of this service for tracing
        """
        self.service_name = service_name
        self._tracer = None

        if OTEL_AVAILABLE:
            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(service_name)

    @property
    def tracer(self):
        """Get the tracer instance."""
        return self._tracer

    @asynccontextmanager
    async def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: str = "internal",
    ) -> AsyncIterator[Span | None]:
        """Create a span context manager.

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

        span_kind = getattr(trace.SpanKind, kind.upper(), trace.SpanKind.INTERNAL)

        with self._tracer.start_as_current_span(name, kind=span_kind) as current_span:
            if attributes:
                for key, value in attributes.items():
                    current_span.set_attribute(key, value)
            try:
                yield current_span
            except Exception as e:
                current_span.set_status(Status(StatusCode.ERROR, str(e)))
                current_span.record_exception(e)
                raise

    def get_current_span(self) -> Span | None:
        """Get the current active span.

        Returns:
            Current Span or None
        """
        if not OTEL_AVAILABLE:
            return None
        return trace.get_current_span()

    def get_trace_context(self) -> tuple[str | None, str | None]:
        """Get current trace_id and span_id.

        Returns:
            Tuple of (trace_id, span_id) or (None, None)
        """
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

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a new trace ID.

        Returns:
            32-character hex trace ID
        """
        return uuid.uuid4().hex[:32]

    @staticmethod
    def generate_span_id() -> str:
        """Generate a new span ID.

        Returns:
            16-character hex span ID
        """
        return uuid.uuid4().hex[:16]

    def inject_context(self, carrier: dict[str, str]) -> dict[str, str]:
        """Inject trace context into a carrier dict (for propagation).

        Args:
            carrier: Dict to inject context into

        Returns:
            Carrier with trace context
        """
        trace_id, span_id = self.get_trace_context()
        if trace_id:
            carrier["trace_id"] = trace_id
        if span_id:
            carrier["span_id"] = span_id
        return carrier

    def extract_context(self, carrier: dict[str, str]) -> Any:
        """Extract trace context from a carrier dict.

        Args:
            carrier: Dict containing trace context

        Returns:
            Context object or None
        """
        if not OTEL_AVAILABLE:
            return None

        # Simple extraction from carrier
        # In production, use proper W3C TraceContext propagation
        trace_id = carrier.get("trace_id")
        span_id = carrier.get("span_id")

        if trace_id and span_id:
            return {"trace_id": trace_id, "span_id": span_id}
        return None

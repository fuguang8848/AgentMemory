"""OpenTelemetry 可观测性模块 — traces + metrics + logs 三大支柱

导出:
    setup_tracing(app_name, endpoint)   初始化 tracer provider
    setup_metrics(app_name, endpoint)    初始化 meter provider
    setup_logging(service_name)          结构化日志 + trace context 注入
    async_trace(name)                    异步函数 span 装饰器
    start_prometheus_server(port, host)  Prometheus HTTP 指标端点

降级策略: OTLP gRPC → ConsoleExporter (当 endpoint 不可用时)
环境变量: OTEL_EXPORTER_OTLP_ENDPOINT
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
from typing import Callable, TypeVar, ParamSpec

# ── tracing ──────────────────────────────────────────────────────────────────
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Status, StatusCode

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _OTLP_TRACE_AVAILABLE = True
except Exception:  # pragma: no cover
    _OTLP_TRACE_AVAILABLE = False
    OTLPSpanExporter = None

# ── metrics ──────────────────────────────────────────────────────────────────
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.metrics import Counter, Histogram

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    _OTLP_METRIC_AVAILABLE = True
except Exception:  # pragma: no cover
    _OTLP_METRIC_AVAILABLE = False
    OTLPMetricExporter = None

# ── logging ─────────────────────────────────────────────────────────────────
from opentelemetry import _logs
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter

try:
    from opentelemetry.sdk.logging import LoggingHandler as OTelLoggingHandler
    _OTEL_LOGGING_HANDLER = True
except Exception:
    OTelLoggingHandler = None
    _OTEL_LOGGING_HANDLER = False

try:
    from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
    _OTLP_LOG_AVAILABLE = True
except Exception:  # pragma: no cover
    _OTLP_LOG_AVAILABLE = False
    OTLPLogExporter = None

# ── context ─────────────────────────────────────────────────────────────────
from opentelemetry.context import attach, detach

__all__ = [
    "setup_tracing",
    "setup_metrics",
    "setup_logging",
    "async_trace",
    "start_prometheus_server",
]

# ────────────────────────────────────────────────────────────────────────────
# Exporter helpers
# ────────────────────────────────────────────────────────────────────────────

def _build_otlp_endpoint(endpoint: str | None) -> str:
    """从 env 或显式参数获取 OTLP endpoint。"""
    return endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")


def _otlp_span_exporter(endpoint: str):
    if _OTLP_TRACE_AVAILABLE and endpoint:
        try:
            return OTLPSpanExporter(endpoint=endpoint, insecure=True)
        except Exception:
            pass
    return ConsoleSpanExporter()


def _otlp_metric_exporter(endpoint: str):
    if _OTLP_METRIC_AVAILABLE and endpoint:
        try:
            return OTLPMetricExporter(endpoint=endpoint, insecure=True)
        except Exception:
            pass
    return ConsoleMetricExporter()


def _otlp_log_exporter(endpoint: str):
    if _OTLP_LOG_AVAILABLE and endpoint:
        try:
            return OTLPLogExporter(endpoint=endpoint, insecure=True)
        except Exception:
            pass
    return ConsoleLogExporter()


# ────────────────────────────────────────────────────────────────────────────
# Public API — setup_tracing
# ────────────────────────────────────────────────────────────────────────────

def setup_tracing(app_name: str, endpoint: str | None = None) -> trace.Tracer:
    """初始化 tracer provider，添加 span processor，导出到 OTLP gRPC。

    参数:
        app_name: 服务名称，作为 tracer 的 instrumentation_scope.name
        endpoint: OTLP gRPC 地址，默认为 http://localhost:4317
                  或从 OTEL_EXPORTER_OTLP_ENDPOINT 环境变量读取

    返回:
        opentelemetry.trace.Tracer 实例

    降级: OTLP 不可用时自动降级到 ConsoleSpanExporter
    """
    endpoint = _build_otlp_endpoint(endpoint)
    exporter = _otlp_span_exporter(endpoint)

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(app_name)
    return tracer


# ────────────────────────────────────────────────────────────────────────────
# Public API — setup_metrics
# ────────────────────────────────────────────────────────────────────────────

# 全局 meter 引用，防止被 GC
_METER: MeterProvider | None = None
_METRICS: dict[str, Counter | Histogram] = {}


def setup_metrics(app_name: str, endpoint: str | None = None) -> metrics.Meter:
    """初始化 meter provider，注册关键指标 counter / histogram。

    关键指标:
        memory_ops_total           Counter  — 内存操作总次数 (label: op)
        pollution_detected_total   Counter  — 污染检测总次数 (label: severity)
        skill_invocations_total    Counter  — 技能调用总次数 (label: skill)
        agent_collaborations_total Counter  — 多 agent 协作总次数 (label: type)
        operation_duration_seconds Histogram — 操作耗时秒数 (label: op)

    参数:
        app_name: 服务名称
        endpoint: OTLP gRPC 地址

    返回:
        opentelemetry.metrics.Meter 实例
    """
    global _METER, _METRICS

    endpoint = _build_otlp_endpoint(endpoint)
    exporter = _otlp_metric_exporter(endpoint)

    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60_000)
    _METER = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(_METER)

    meter = metrics.get_meter(app_name)

    # Counter: memory_ops_total
    _METRICS["memory_ops_total"] = meter.create_counter(
        name="memory_ops_total",
        description="Total memory operations",
        unit="1",
    )

    # Counter: pollution_detected_total
    _METRICS["pollution_detected_total"] = meter.create_counter(
        name="pollution_detected_total",
        description="Total pollution detection events",
        unit="1",
    )

    # Counter: skill_invocations_total
    _METRICS["skill_invocations_total"] = meter.create_counter(
        name="skill_invocations_total",
        description="Total skill invocations",
        unit="1",
    )

    # Counter: agent_collaborations_total
    _METRICS["agent_collaborations_total"] = meter.create_counter(
        name="agent_collaborations_total",
        description="Total agent collaboration events",
        unit="1",
    )

    # Histogram: operation_duration_seconds
    _METRICS["operation_duration_seconds"] = meter.create_histogram(
        name="operation_duration_seconds",
        description="Duration of operations in seconds",
        unit="s",
    )

    return meter


def get_metric(name: str) -> Counter | Histogram | None:
    """获取已注册的指标。"""
    return _METRICS.get(name)


# ────────────────────────────────────────────────────────────────────────────
# Public API — setup_logging
# ────────────────────────────────────────────────────────────────────────────

_LOGGER_PROVIDER: LoggerProvider | None = None


def setup_logging(service_name: str, endpoint: str | None = None) -> logging.Logger:
    """配置结构化日志，自动注入 trace context (trace_id, span_id)。

    参数:
        service_name: 服务名称
        endpoint: OTLP gRPC 地址（用于日志导出）

    返回:
        配置好的 logging.Logger
    """
    global _LOGGER_PROVIDER

    endpoint = _build_otlp_endpoint(endpoint)
    exporter = _otlp_log_exporter(endpoint)

    _LOGGER_PROVIDER = LoggerProvider()
    _LOGGER_PROVIDER.add_log_record_processor(BatchLogRecordProcessor(exporter))
    _logs.set_logger_provider(_LOGGER_PROVIDER)

    # 用 OTel LoggingHandler 桥接 stdlib logging → OTel SDK
    if OTelLoggingHandler is not None:
        otel_handler = OTelLoggingHandler()
        otel_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] "
            "trace_id=%(otelTraceID)s span_id=%(otelSpanID)s "
            "%(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        otel_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers.clear()
        root_logger.addHandler(otel_handler)
    else:
        # Fallback: 简单控制台日志，不做 OTel 集成
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    # 抑制 noisy 第三方库的废话日志
    for noisy in ["httpx", "urllib3", "grpc"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger()


# ────────────────────────────────────────────────────────────────────────────
# Decorator — async_trace
# ────────────────────────────────────────────────────────────────────────────

P = ParamSpec("P")
T = TypeVar("T")


def async_trace(name: str | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """异步函数 span 装饰器 — 为任何 async def 函数添加 span，自动记录异常。

    用法:
        @async_trace("my_operation")
        async def do_something(x: int) -> str:
            ...

    参数:
        name: span 名称，默认为函数 __qualname__
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = trace.get_tracer(func.__module__)
            with tracer.start_as_current_span(span_name) as span:
                try:
                    span.set_attribute("function", func.__qualname__)
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:  # noqa: BLE001
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)
                    raise

        return wrapper  # type: ignore[return-value]
    return decorator


# ----------------------------------------------------------------
# II7: Prometheus HTTP metrics endpoint
# ----------------------------------------------------------------

import http.server
import threading
import socketserver
from prometheus_client import (
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    Gauge,
    Counter as PromCounter,
    Histogram as PromHistogram,
)
from prometheus_client.openmetrics.exposition import generate_latest as om_generate_latest


class _PrometheusRegistry:
    """Prometheus metrics registry - exposes spectrai_* metrics at /metrics endpoint."""

    def __init__(self):
        self._registry = CollectorRegistry()
        self.pollution_events = PromCounter(
            "spectrai_pollution_events_total",
            "Total pollution detection events",
            ["severity", "source"],
            registry=self._registry,
        )
        self.collaboration_duration = PromHistogram(
            "spectrai_collaboration_duration_seconds",
            "Agent collaboration duration in seconds",
            ["team", "mode"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self._registry,
        )
        self.active_spans = Gauge(
            "spectrai_active_spans",
            "Number of currently active spans",
            ["service"],
            registry=self._registry,
        )

    def emit_pollution(self, severity: str, source: str = "sanitizer") -> None:
        self.pollution_events.labels(severity=severity, source=source).inc()

    def emit_collaboration_duration(self, team: str, mode: str, duration: float) -> None:
        self.collaboration_duration.labels(team=team, mode=mode).observe(duration)

    def set_active_spans(self, service: str, count: int) -> None:
        self.active_spans.labels(service=service).set(count)

    def generate(self) -> bytes:
        return generate_latest(self._registry)


_prom_registry = None
_prom_server_thread = None
_prom_shutdown = threading.Event()


def get_prometheus_registry():
    global _prom_registry
    if _prom_registry is None:
        _prom_registry = _PrometheusRegistry()
    return _prom_registry


class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path in ("/metrics", "/metrics/prometheus"):
            reg = get_prometheus_registry()
            output = reg.generate()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_prometheus_server(port=9090, host="0.0.0.0", start_span_observer=True):
    """Start Prometheus HTTP metrics endpoint (background thread).

    Exposes:
      GET /metrics            Prometheus text format
      GET /metrics/prometheus same
      GET /health             liveness probe

    Custom metrics:
      spectrai_pollution_events_total         Counter (severity, source)
      spectrai_active_spans                  Gauge (service)
      spectrai_collaboration_duration_seconds Histogram (team, mode)

    Returns:
        Endpoint URL string, e.g. "http://0.0.0.0:9090/metrics"

    Usage:
        >>> from agentmemory.extensions.v2 import (
        ...     setup_tracing, setup_metrics, start_prometheus_server
        ... )
        >>> tracer = setup_tracing("AgentMemory")
        >>> meter = setup_metrics("AgentMemory")
        >>> endpoint = start_prometheus_server(port=9090)
    """
    global _prom_server_thread, _prom_shutdown

    if _prom_server_thread is not None and _prom_server_thread.is_alive():
        return f"http://{host}:{port}/metrics"

    _prom_shutdown.clear()

    def run_server():
        try:
            with _ThreadingHTTPServer((host, port), _MetricsHandler) as httpd:
                httpd.server_name = "spectrai-metrics"
                while not _prom_shutdown.is_set():
                    httpd.handle_request()
        except Exception:
            pass

    _prom_server_thread = threading.Thread(
        target=run_server, daemon=True, name="prom-metrics"
    )
    _prom_server_thread.start()

    endpoint = f"http://{host}:{port}/metrics"

    if start_span_observer:
        def observe_spans():
            import time as _time
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider

            while not _prom_shutdown.is_set():
                try:
                    provider = trace.get_tracer_provider()
                    if isinstance(provider, TracerProvider):
                        reg = get_prometheus_registry()
                        reg.set_active_spans("AgentMemory", 0)
                except Exception:
                    pass
                _time.sleep(5)

        _obs = threading.Thread(target=observe_spans, daemon=True, name="span-observer")
        _obs.start()

    return endpoint

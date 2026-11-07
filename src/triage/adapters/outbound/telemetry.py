"""OpenTelemetry adapter. Exports traces over OTLP and metrics to Prometheus and OTLP.

Setting the standard OTEL_EXPORTER_OTLP_* variables is all that is needed to send traces to a
Collector or directly to Langfuse. ADK spans use the same global tracer provider.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_telemetry(service_name: str = "arabic-complaint-triage") -> None:
    resource = Resource.create({"service.name": service_name})
    otlp_enabled = bool(
        os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )

    tracer_provider = TracerProvider(resource=resource)
    if otlp_enabled:
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    readers: list[Any] = [PrometheusMetricReader()]
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))


class OtelTelemetry:
    def __init__(self) -> None:
        self._tracer = trace.get_tracer("triage")
        self._meter = metrics.get_meter("triage")
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}

    def span(self, name: str, **attributes: Any) -> AbstractContextManager[Any]:
        return self._tracer.start_as_current_span(name, attributes=attributes)

    def count(self, name: str, value: int = 1, **attributes: Any) -> None:
        counter = self._counters.get(name)
        if counter is None:
            counter = self._counters[name] = self._meter.create_counter(name)
        counter.add(value, attributes)

    def record(self, name: str, value: float, **attributes: Any) -> None:
        histogram = self._histograms.get(name)
        if histogram is None:
            histogram = self._histograms[name] = self._meter.create_histogram(name)
        histogram.record(value, attributes)

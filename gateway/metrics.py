"""Prometheus metrics. Each gateway app gets its own registry so tests stay isolated."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter("gateway_requests_total", "Chat requests handled",
                                ["model", "provider", "status", "cache"], registry=self.registry)
        self.latency = Histogram("gateway_request_latency_seconds", "End-to-end request latency",
                                 ["model", "cache"], buckets=LATENCY_BUCKETS, registry=self.registry)
        self.upstream_latency = Histogram("gateway_upstream_latency_seconds", "Upstream provider latency",
                                          ["provider", "model", "outcome"], buckets=LATENCY_BUCKETS,
                                          registry=self.registry)
        self.tokens = Counter("gateway_tokens_total", "Tokens processed", ["model", "kind"], registry=self.registry)
        self.cost = Counter("gateway_cost_usd_total", "Estimated spend in USD", ["model"], registry=self.registry)
        self.saved = Counter("gateway_cache_saved_usd_total", "Estimated USD saved by cache hits",
                             registry=self.registry)
        self.fallbacks = Counter("gateway_fallbacks_total", "Times a request fell back to another model",
                                 ["from_model", "to_model"], registry=self.registry)
        self.rate_limited = Counter("gateway_rate_limited_total", "Requests rejected by rate limiting",
                                    ["key"], registry=self.registry)
        self.circuit_open = Gauge("gateway_circuit_open", "1 if the model's circuit breaker is open",
                                  ["model"], registry=self.registry)
        self.cache_entries = Gauge("gateway_cache_entries", "Entries in the semantic cache", registry=self.registry)

"""Prometheus metrics for the OpenAI-compatible proxy."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Hashable

from guarded_hotshard.metrics import percentile


def build_proxy_metrics(mode_name: str, version: str) -> ProxyMetrics:
    try:
        import prometheus_client  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Prometheus metrics require prometheus_client. "
            "Install with: pip install 'guarded-hotshard[server]'"
        ) from e
    return ProxyMetrics(mode_name=mode_name, version=version)


class ProxyMetrics:
    """Per-process metrics registry (one instance per FastAPI app)."""

    def __init__(self, *, mode_name: str, version: str) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Info

        self.registry: CollectorRegistry = CollectorRegistry()
        self._max_tenants = 256
        self._max_samples = 200
        self._tenant_samples: OrderedDict[str, deque[float]] = OrderedDict()

        Info(
            "ghs_proxy",
            "guarded-hotshard proxy build info",
            registry=self.registry,
        ).info({"version": version, "mode": mode_name})

        self.queue_depth = Gauge(
            "ghs_scheduler_queue_depth",
            "Number of requests waiting in the priority queue",
            registry=self.registry,
        )
        self.scheduler_in_flight = Gauge(
            "ghs_scheduler_in_flight",
            "Requests dispatched to the backend but not yet completed",
            registry=self.registry,
        )
        self.tenant_wall_p99_seconds = Gauge(
            "ghs_tenant_wall_latency_p99_seconds",
            "Rolling p99 end-to-end wall latency per tenant (seconds)",
            ("tenant",),
            registry=self.registry,
        )
        self.proxy_requests_total = Counter(
            "ghs_proxy_requests_total",
            "Ingress requests handled by the proxy",
            ("path", "stream", "storm"),
            registry=self.registry,
        )
        self.backend_http_total = Counter(
            "ghs_backend_http_requests_total",
            "Outbound HTTP calls to the upstream OpenAI-compatible backend",
            ("role",),
            registry=self.registry,
        )
        self.tmr_launches_total = Counter(
            "ghs_tmr_parallel_launches_total",
            "Times a redundant parallel backend call was started (TMR path)",
            registry=self.registry,
        )
        self.protected_lane_tmr_total = Counter(
            "ghs_protected_lane_redundancy_activations_total",
            "TMR activations while running in protected_lane mode",
            registry=self.registry,
        )
        self.storm_requests_total = Counter(
            "ghs_storm_like_requests_total",
            "Requests tagged as storm-like (header X-GHS-Storm or --storm-users match)",
            registry=self.registry,
        )

    def sync_queue_gauges(self, queue_depth: int, in_flight: int) -> None:
        self.queue_depth.set(float(queue_depth))
        self.scheduler_in_flight.set(float(in_flight))

    def record_completion(
        self,
        *,
        tenant: Hashable,
        wall_seconds: float,
        path: str,
        stream: bool,
        storm: bool,
        backend_primary: int,
        backend_tmr: int,
        protected_lane_tmr: bool,
    ) -> None:
        tkey = str(tenant)
        if tkey not in self._tenant_samples:
            if len(self._tenant_samples) >= self._max_tenants:
                self._tenant_samples.popitem(last=False)
            self._tenant_samples[tkey] = deque(maxlen=self._max_samples)
        self._tenant_samples[tkey].append(wall_seconds)
        self._tenant_samples.move_to_end(tkey)
        samples = list(self._tenant_samples[tkey])
        p99 = percentile(samples, 0.99)
        self.tenant_wall_p99_seconds.labels(tenant=tkey).set(p99)

        self.proxy_requests_total.labels(
            path=path,
            stream="true" if stream else "false",
            storm="true" if storm else "false",
        ).inc()
        if storm:
            self.storm_requests_total.inc()

        for _ in range(backend_primary):
            self.backend_http_total.labels(role="primary").inc()
        for _ in range(backend_tmr):
            self.backend_http_total.labels(role="tmr").inc()
        if backend_tmr > 0:
            self.tmr_launches_total.inc()
        if protected_lane_tmr:
            self.protected_lane_tmr_total.inc()

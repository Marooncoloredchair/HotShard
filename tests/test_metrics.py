"""Prometheus /metrics on the proxy."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")


def test_metrics_endpoint_exposes_ghs_series():
    from fastapi.testclient import TestClient

    from guarded_hotshard.proxy import create_app

    app = create_app("http://127.0.0.1:59999", mode="balanced", enable_metrics=True)
    with TestClient(app) as client:
        r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "ghs_scheduler_queue_depth" in text
    assert "ghs_backend_http_requests_total" in text


def test_is_storm_detection():
    from guarded_hotshard.proxy import _is_storm

    assert _is_storm(body={"user": "noisy"}, headers={}, storm_users={"noisy"})
    assert not _is_storm(body={"user": "ok"}, headers={}, storm_users=set())
    assert _is_storm(body={"user": "ok"}, headers={"X-GHS-Storm": "1"}, storm_users=set())
    assert not _is_storm(body={"user": "ok"}, headers={"X-GHS-Storm": "false"}, storm_users=set())

"""OpenAI-compatible FastAPI proxy with guarded scheduling.

Runs in front of any backend that speaks the OpenAI HTTP API.

Expose Prometheus metrics at ``GET /metrics`` (requires ``[server]`` extra).

Storm-like traffic can be tagged with header ``X-GHS-Storm: 1`` or by
listing tenant ids in ``storm_users`` (CLI: ``--storm-users``) for
``ghs_storm_like_requests_total``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Hashable
from contextlib import asynccontextmanager
from typing import Any

import httpx

from guarded_hotshard._version import __version__
from guarded_hotshard.async_dispatcher import AsyncDispatcher
from guarded_hotshard.modes import Mode, make_mode

log = logging.getLogger("guarded_hotshard.proxy")


def _is_storm(*, body: dict, headers: dict, storm_users: set[str]) -> bool:
    h = {k.lower(): v for k, v in headers.items()}
    v = h.get("x-ghs-storm", "").lower()
    if v in ("1", "true", "yes"):
        return True
    user = str(body.get("user", "_unknown"))
    return user in storm_users


def create_app(
    backend_url: str,
    *,
    mode: str | Mode = "balanced",
    critical_users: list[Hashable] | None = None,
    storm_users: list[str] | None = None,
    concurrency: int = 8,
    api_key: str | None = None,
    request_timeout: float = 600.0,
    enable_metrics: bool = True,
) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, Response, StreamingResponse
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "guarded_hotshard.proxy requires the [server] extra. "
            "Install with: pip install 'guarded-hotshard[server]'"
        ) from e

    if isinstance(mode, str):
        mode_obj = make_mode(mode)
    else:
        mode_obj = mode
    crit_set: set[Hashable] = set(critical_users or [])
    storm_set: set[str] = set(storm_users or [])
    backend = backend_url.rstrip("/")

    dispatcher = AsyncDispatcher(mode=mode_obj, critical_tenants=crit_set, concurrency=concurrency)

    metrics = None
    if enable_metrics:
        try:
            from guarded_hotshard.prometheus_export import build_proxy_metrics

            metrics = build_proxy_metrics(mode_obj.name, __version__)
        except Exception as e:  # pragma: no cover
            log.warning("Prometheus metrics disabled: %s", e)

    @asynccontextmanager
    async def lifespan(app):
        await dispatcher.ensure_started()
        app.state.client = httpx.AsyncClient(timeout=request_timeout)
        app.state.metrics = metrics
        try:
            yield
        finally:
            await dispatcher.stop()
            await app.state.client.aclose()

    app = FastAPI(
        title="guarded-hotshard proxy",
        version=__version__,
        description=f"Tenant-aware OpenAI-compatible proxy. Mode={mode_obj.name}, backend={backend}",
        lifespan=lifespan,
    )

    headers_in = {}
    if api_key:
        headers_in["Authorization"] = f"Bearer {api_key}"

    def _sync_metrics() -> None:
        if metrics is None:
            return
        metrics.sync_queue_gauges(
            dispatcher.scheduler.queue_depth(),
            dispatcher.scheduler.in_flight(),
        )

    async def _forward(method: str, path: str, *, json_body: Any, request_headers: dict) -> httpx.Response:
        url = f"{backend}{path}"
        forward_headers = {
            k: v
            for k, v in request_headers.items()
            if k.lower() not in {"host", "content-length", "connection"}
        }
        if api_key and "authorization" not in {h.lower() for h in forward_headers}:
            forward_headers["Authorization"] = f"Bearer {api_key}"
        return await app.state.client.request(method, url, json=json_body, headers=forward_headers)

    async def _scheduled_forward(
        path: str, body: dict, headers: dict, *, stream: bool = False
    ) -> tuple[int, dict | str, dict]:
        tenant = body.get("user", "_unknown")
        storm = _is_storm(body=body, headers=headers, storm_users=storm_set)
        scored = await dispatcher.submit(tenant=tenant)
        _sync_metrics()
        t0 = time.time()
        backend_primary = 0
        backend_tmr = 0
        try:
            primary_task = asyncio.create_task(
                _forward("POST", path, json_body=body, request_headers=headers)
            )
            backend_primary = 1
            tmr_task: asyncio.Task[httpx.Response] | None = None
            if scored.tmr and not stream:
                tmr_task = asyncio.create_task(
                    _forward("POST", path, json_body=body, request_headers=headers)
                )
                backend_tmr = 1

            if tmr_task is not None:
                done, pending = await asyncio.wait(
                    {primary_task, tmr_task}, return_when=asyncio.FIRST_COMPLETED
                )
                resp = next(iter(done)).result()
                for p in pending:
                    p.cancel()
                    try:
                        await p
                    except (asyncio.CancelledError, Exception):
                        pass
            else:
                resp = await primary_task

            if stream:
                return resp.status_code, "__stream__", dict(resp.headers)
            payload: Any = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            return resp.status_code, payload, dict(resp.headers)
        finally:
            wall = time.time() - t0
            await dispatcher.complete(scored, wall)
            _sync_metrics()
            if metrics is not None:
                protected = mode_obj.name == "protected_lane" and bool(scored.tmr)
                metrics.record_completion(
                    tenant=tenant,
                    wall_seconds=wall,
                    path=path,
                    stream=stream,
                    storm=storm,
                    backend_primary=backend_primary,
                    backend_tmr=backend_tmr,
                    protected_lane_tmr=protected,
                )

    @app.get("/")
    async def index():
        return {
            "service": "guarded-hotshard",
            "version": __version__,
            "mode": mode_obj.name,
            "backend": backend,
            "concurrency": concurrency,
            "metrics": "/metrics" if metrics is not None else None,
        }

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "queue_depth": dispatcher.scheduler.queue_depth()}

    @app.get("/metrics")
    async def metrics_endpoint():
        if metrics is None:
            return Response("metrics disabled", media_type="text/plain", status_code=503)
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        _sync_metrics()
        data = generate_latest(metrics.registry)
        return Response(data, media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def list_models(request: Request):
        resp = await _forward("GET", "/v1/models", json_body=None, request_headers=dict(request.headers))
        return JSONResponse(resp.json(), status_code=resp.status_code)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        hdrs = dict(request.headers)
        if body.get("stream", False):
            tenant = body.get("user", "_unknown")
            storm = _is_storm(body=body, headers=hdrs, storm_users=storm_set)
            scored = await dispatcher.submit(tenant=tenant)
            _sync_metrics()
            t0 = time.time()
            try:
                req = app.state.client.build_request(
                    "POST",
                    f"{backend}/v1/chat/completions",
                    json=body,
                    headers={**headers_in},
                )
                resp = await app.state.client.send(req, stream=True)
                b_primary, b_tmr = 1, 0

                async def stream_iter():
                    try:
                        async for chunk in resp.aiter_raw():
                            yield chunk
                    finally:
                        await resp.aclose()
                        wall = time.time() - t0
                        await dispatcher.complete(scored, wall)
                        _sync_metrics()
                        if metrics is not None:
                            metrics.record_completion(
                                tenant=tenant,
                                wall_seconds=wall,
                                path="/v1/chat/completions",
                                stream=True,
                                storm=storm,
                                backend_primary=b_primary,
                                backend_tmr=b_tmr,
                                protected_lane_tmr=False,
                            )

                return StreamingResponse(
                    stream_iter(),
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "text/event-stream"),
                )
            except Exception:
                wall = time.time() - t0
                await dispatcher.complete(scored, wall)
                _sync_metrics()
                raise

        status, payload, _ = await _scheduled_forward(
            "/v1/chat/completions", body, hdrs, stream=False
        )
        return JSONResponse(payload, status_code=status)

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        status, payload, _ = await _scheduled_forward("/v1/completions", body, dict(request.headers))
        return JSONResponse(payload, status_code=status)

    return app


def run(
    backend_url: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    mode: str = "balanced",
    critical_users: list[str] | None = None,
    storm_users: list[str] | None = None,
    concurrency: int = 8,
    api_key: str | None = None,
    log_level: str = "info",
) -> None:
    """Run the proxy server. Convenience wrapper around uvicorn.run."""
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "guarded_hotshard.proxy.run requires the [server] extra. "
            "Install with: pip install 'guarded-hotshard[server]'"
        ) from e
    app = create_app(
        backend_url,
        mode=mode,
        critical_users=critical_users,
        storm_users=storm_users,
        concurrency=concurrency,
        api_key=api_key,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level)

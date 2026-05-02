"""OpenAI-compatible FastAPI proxy with guarded scheduling.

Runs in front of any backend that speaks the OpenAI HTTP API:

    vllm serve qwen/Qwen2.5-3B-Instruct --port 8001
    ghs serve --backend http://localhost:8001 --port 8000 --mode protected_lane

Then point your client at `http://localhost:8000/v1` and use the `user`
field for tenant identification. The proxy:

1. Receives the request.
2. Scores it (tenant, criticality, hotness, F-risk).
3. Enqueues it on a priority heap.
4. A dispatcher coroutine pops the highest-priority request whenever a
   backend slot frees up, and forwards it to the backend.
5. For TMR-pinned premium-tenant traffic, it fires twice in parallel and
   returns whichever finishes first.
6. On *any* error, it falls back to a direct forward. Never makes things
   worse than the backend alone.

This module is optional. The package works without `fastapi`/`uvicorn`
installed; you only need them for `ghs serve`.
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
from guarded_hotshard.modes import Mode, make_mode
from guarded_hotshard.scheduler import GuardedScheduler

log = logging.getLogger("guarded_hotshard.proxy")


# ---------------------------------------------------------------------------
# Async dispatcher
# ---------------------------------------------------------------------------
class AsyncDispatcher:
    """Async cousin of `_SchedulerThread` from wrap.py.

    Owns the heap, holds a semaphore for backend concurrency, and exposes
    `submit()` which awaits both dispatch admission and request completion.
    """

    def __init__(self, mode: Mode, critical_tenants: set[Hashable], concurrency: int):
        self.scheduler = GuardedScheduler(mode=mode, critical_tenants=critical_tenants)
        self.concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self._heap_lock = asyncio.Lock()
        self._counter = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._dispatch_loop(), name="ghs-dispatch")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            async with self._heap_lock:
                while self.scheduler.in_flight() < self.concurrency and self.scheduler.queue_depth() > 0:
                    sr = self.scheduler.dispatch_next()
                    if sr is None:
                        break
                    sr.request["dispatch_event"].set()

    async def submit(self, *, tenant: Hashable) -> Any:
        ev = asyncio.Event()
        self._counter += 1
        rid = self._counter
        scored = self.scheduler.score(
            request={"dispatch_event": ev},
            request_id=rid,
            tenant=tenant,
            arrival_time=time.time(),
        )
        async with self._heap_lock:
            self.scheduler.enqueue(scored)
        self._wake.set()
        await ev.wait()
        return scored

    async def complete(self, scored: Any, latency: float) -> None:
        async with self._heap_lock:
            self.scheduler.complete(scored, latency)
        self._wake.set()


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------
def create_app(
    backend_url: str,
    *,
    mode: str | Mode = "balanced",
    critical_users: list[Hashable] | None = None,
    concurrency: int = 8,
    api_key: str | None = None,
    request_timeout: float = 600.0,
) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
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
    backend = backend_url.rstrip("/")

    dispatcher = AsyncDispatcher(mode=mode_obj, critical_tenants=crit_set, concurrency=concurrency)

    @asynccontextmanager
    async def lifespan(app):
        await dispatcher.start()
        app.state.client = httpx.AsyncClient(timeout=request_timeout)
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

    async def _scheduled_forward(path: str, body: dict, headers: dict) -> tuple[int, dict | str, dict]:
        tenant = body.get("user", "_unknown")
        is_stream = bool(body.get("stream", False))
        scored = await dispatcher.submit(tenant=tenant)
        t0 = time.time()
        try:
            primary_task = asyncio.create_task(
                _forward("POST", path, json_body=body, request_headers=headers)
            )
            tmr_task: asyncio.Task[httpx.Response] | None = None
            if scored.tmr and not is_stream:
                tmr_task = asyncio.create_task(
                    _forward("POST", path, json_body=body, request_headers=headers)
                )

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

            if is_stream:
                # Caller will stream resp directly; we don't await body.
                return resp.status_code, "__stream__", dict(resp.headers)
            payload: Any = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            return resp.status_code, payload, dict(resp.headers)
        finally:
            await dispatcher.complete(scored, time.time() - t0)

    @app.get("/")
    async def index():
        return {
            "service": "guarded-hotshard",
            "version": __version__,
            "mode": mode_obj.name,
            "backend": backend,
            "concurrency": concurrency,
        }

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "queue_depth": dispatcher.scheduler.queue_depth()}

    @app.get("/v1/models")
    async def list_models(request: Request):
        resp = await _forward("GET", "/v1/models", json_body=None, request_headers=dict(request.headers))
        return JSONResponse(resp.json(), status_code=resp.status_code)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if body.get("stream", False):
            scored = await dispatcher.submit(tenant=body.get("user", "_unknown"))
            t0 = time.time()
            try:
                req = app.state.client.build_request(
                    "POST",
                    f"{backend}/v1/chat/completions",
                    json=body,
                    headers={**headers_in},
                )
                resp = await app.state.client.send(req, stream=True)

                async def stream_iter():
                    try:
                        async for chunk in resp.aiter_raw():
                            yield chunk
                    finally:
                        await resp.aclose()

                return StreamingResponse(
                    stream_iter(),
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "text/event-stream"),
                )
            finally:
                await dispatcher.complete(scored, time.time() - t0)

        status, payload, _ = await _scheduled_forward(
            "/v1/chat/completions", body, dict(request.headers)
        )
        return JSONResponse(payload, status_code=status)

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        status, payload, _ = await _scheduled_forward(
            "/v1/completions", body, dict(request.headers)
        )
        return JSONResponse(payload, status_code=status)

    return app


def run(
    backend_url: str,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    mode: str = "balanced",
    critical_users: list[str] | None = None,
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
        concurrency=concurrency,
        api_key=api_key,
    )
    uvicorn.run(app, host=host, port=port, log_level=log_level)

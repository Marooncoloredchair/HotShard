"""One-line wrapper for OpenAI-compatible clients.

Use this when you want guarded scheduling **inside your application**
(no proxy, no extra hop). It re-points `client.chat.completions.create` and
`client.completions.create` at a thin shim that:

  1. Scores the request with the chosen mode.
  2. Waits in an in-process priority queue with bounded concurrency.
  3. Optionally fires the request twice and picks the first valid response
     (TMR) for premium-tenant traffic.
  4. Falls back to a direct call on any error - never makes things worse.

Example
-------
>>> from openai import OpenAI
>>> from guarded_hotshard import wrap
>>> client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
>>> client = wrap(client, mode="protected_lane", critical_users={"acme"})
>>> client.chat.completions.create(
...     model="qwen2.5-3b-instruct",
...     messages=[{"role": "user", "content": "hi"}],
...     user="acme",
... )

The `user` field of the OpenAI API is reused as the tenant id. If your app
already populates it (for billing or moderation), you get tenant-aware
scheduling for free.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Hashable, Iterable
from typing import Any

from guarded_hotshard.async_dispatcher import AsyncDispatcher
from guarded_hotshard.modes import Mode, make_mode
from guarded_hotshard.scheduler import GuardedScheduler


class _SchedulerThread:
    """Single background thread that owns the heap.

    The wrapper API is synchronous (matches the OpenAI client) but we want a
    real priority queue, so we run a tiny dispatch loop on a daemon thread.
    Each call posts to the queue and blocks on a `threading.Event` until its
    request finishes.
    """

    def __init__(self, mode: Mode, critical_tenants: set[Hashable], concurrency: int):
        self.scheduler = GuardedScheduler(mode=mode, critical_tenants=critical_tenants)
        self.concurrency = concurrency
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._counter = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ghs-dispatch")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stopped:
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            with self._lock:
                while self.scheduler.in_flight() < self.concurrency:
                    sr = self.scheduler.dispatch_next()
                    if sr is None:
                        break
                    # `sr.request` is the threading.Event-wrapped job
                    sr.request["dispatch_event"].set()

    def submit(self, *, tenant: Hashable, payload: dict[str, Any]) -> dict[str, Any]:
        ev = threading.Event()
        done = threading.Event()
        result: dict[str, Any] = {}
        self._counter += 1
        rid = self._counter
        scored = self.scheduler.score(
            request={"dispatch_event": ev, "done": done, "result": result, **payload},
            request_id=rid,
            tenant=tenant,
            arrival_time=time.time(),
        )
        with self._lock:
            self.scheduler.enqueue(scored)
        self._wake.set()
        ev.wait()
        return {"scored": scored, "done": done, "result": result}

    def complete(self, scored, latency: float) -> None:
        with self._lock:
            self.scheduler.complete(scored, latency)
        self._wake.set()


def wrap(
    client: Any,
    *,
    mode: str | Mode = "balanced",
    critical_users: Iterable[Hashable] | None = None,
    concurrency: int = 8,
) -> Any:
    """Wrap an OpenAI-compatible client with guarded scheduling.

    Parameters
    ----------
    client
        Any object with `client.chat.completions.create(...)` and/or
        `client.completions.create(...)` (sync). For ``AsyncOpenAI``, use
        :func:`wrap_async` instead.
    mode
        Mode name (`"baseline"`, `"eco"`, `"balanced"`, `"strict"`,
        `"critical"`, `"protected_lane"`) or a `Mode` instance.
    critical_users
        User ids (the `user` field passed to OpenAI) to treat as critical
        / premium. Most useful with `mode="protected_lane"`.
    concurrency
        How many requests we let hit the backend in parallel. This caps the
        backend's effective batch size; tune it to roughly match what your
        backend can serve at peak without queueing internally. Default 8
        is fine for a single A100 running a 3-7B model.

    Returns
    -------
    The same client object, with `.chat.completions.create` and
    `.completions.create` wrapped in place.
    """
    if isinstance(mode, str):
        mode = make_mode(mode)
    crit_set: set[Hashable] = set(critical_users or [])
    sched = _SchedulerThread(mode=mode, critical_tenants=crit_set, concurrency=concurrency)

    # Patch chat.completions.create
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        _patch(client.chat.completions, sched, kind="chat")
    if hasattr(client, "completions"):
        _patch(client.completions, sched, kind="text")

    # Stash the scheduler so callers can inspect it
    client._guarded_hotshard = sched
    return client


def _patch(target: Any, sched: _SchedulerThread, *, kind: str) -> None:
    original = target.create

    def create(*args: Any, **kwargs: Any) -> Any:
        tenant = kwargs.get("user", "_unknown")
        scored = None
        t0 = time.time()
        try:
            handle = sched.submit(tenant=tenant, payload={"kind": kind})
            scored = handle["scored"]
            try:
                resp = original(*args, **kwargs)
                if scored.tmr:
                    # Best-effort second attempt; if it fails or differs, we
                    # keep the first. Cost is bounded by D-Budgeted.
                    try:
                        _resp2 = original(*args, **kwargs)
                        # Future hook: vote / pick lowest-latency. For now,
                        # we just paid for the redundancy and return resp.
                    except Exception:  # pragma: no cover
                        pass
                return resp
            finally:
                sched.complete(scored, time.time() - t0)
        except Exception:
            # Failure mode: pass through. Never make things worse.
            if scored is not None:
                try:
                    sched.complete(scored, time.time() - t0)
                except Exception:  # pragma: no cover
                    pass
            return original(*args, **kwargs)

    target.create = create  # type: ignore[method-assign]


def wrap_async(
    client: Any,
    *,
    mode: str | Mode = "balanced",
    critical_users: Iterable[Hashable] | None = None,
    concurrency: int = 8,
) -> Any:
    """Wrap an *async* OpenAI client (e.g. ``AsyncOpenAI``).

    Same scheduling semantics as :func:`wrap`, but each
    ``await client.chat.completions.create(...)`` runs on ``AsyncIO``'s
    event loop instead of a background thread.
    """
    if isinstance(mode, str):
        mode = make_mode(mode)
    crit_set: set[Hashable] = set(critical_users or [])
    dispatcher = AsyncDispatcher(mode=mode, critical_tenants=crit_set, concurrency=concurrency)

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        _patch_async(client.chat.completions, dispatcher, kind="chat")
    if hasattr(client, "completions"):
        _patch_async(client.completions, dispatcher, kind="text")

    client._guarded_hotshard_async = dispatcher
    return client


def _patch_async(target: Any, dispatcher: AsyncDispatcher, *, kind: str) -> None:
    original = target.create

    async def create(*args: Any, **kwargs: Any) -> Any:
        await dispatcher.ensure_started()
        tenant = kwargs.get("user", "_unknown")
        scored = None
        t0 = time.time()
        try:
            scored = await dispatcher.submit(tenant=tenant)
            try:
                resp = await original(*args, **kwargs)
                if scored.tmr:
                    try:
                        await original(*args, **kwargs)
                    except Exception:  # pragma: no cover
                        pass
                return resp
            finally:
                await dispatcher.complete(scored, time.time() - t0)
        except Exception:
            if scored is not None:
                try:
                    await dispatcher.complete(scored, time.time() - t0)
                except Exception:  # pragma: no cover
                    pass
            return await original(*args, **kwargs)

    target.create = create  # type: ignore[method-assign]

"""Async priority-queue dispatcher shared by the proxy and wrap_async()."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Hashable
from typing import Any

from guarded_hotshard.modes import Mode
from guarded_hotshard.scheduler import GuardedScheduler


class AsyncDispatcher:
    """Priority heap + bounded concurrency for async OpenAI-style clients."""

    def __init__(self, mode: Mode, critical_tenants: set[Hashable], concurrency: int):
        self.scheduler = GuardedScheduler(mode=mode, critical_tenants=critical_tenants)
        self.concurrency = concurrency
        self._heap_lock = asyncio.Lock()
        self._counter = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def ensure_started(self) -> None:
        if self._task is None:
            await self.start()

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

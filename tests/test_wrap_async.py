"""Async client wrapper."""

from __future__ import annotations

import pytest

from guarded_hotshard import wrap_async


@pytest.mark.asyncio
async def test_wrap_async_runs_create():
    calls: list[str] = []

    class Completions:
        async def create(self, *a, **kw):
            calls.append(kw.get("user", ""))
            return {"choices": [{"message": {"content": "ok"}}]}

    class Chat:
        def __init__(self):
            self.completions = Completions()

    class Dummy:
        def __init__(self):
            self.chat = Chat()

    c = wrap_async(Dummy(), mode="baseline", concurrency=2)
    out = await c.chat.completions.create(model="m", messages=[], user="tenant-a")
    assert out["choices"][0]["message"]["content"] == "ok"
    assert calls == ["tenant-a"]

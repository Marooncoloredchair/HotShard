"""End-to-end test: run the demo against an in-process mock OpenAI backend.

We spin up a tiny FastAPI app on a free port that returns deterministic
responses with a small delay, then ask `run_demo_async` to hit it. This
verifies the full pipeline (workload -> scheduler -> async dispatch -> HTTP
forward -> metrics -> JSON) without needing a real GPU.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import contextmanager

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def _mock_backend():
    """Run a tiny chat-completions backend on a free port."""
    import uvicorn
    from fastapi import FastAPI, Request

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        msgs = body.get("messages", [])
        last = msgs[-1]["content"] if msgs else ""
        # Add a small per-request delay so the priority order matters.
        await asyncio.sleep(0.05)
        text = f"echo: {last[:30]}"
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        }

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve())

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    # Wait for the server to come up
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            s.close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        pytest.fail("mock backend never started")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_demo_against_mock_backend(tmp_path):
    from guarded_hotshard.demo import run_demo_async

    with _mock_backend() as backend_url:
        result = await run_demo_async(
            backend_url=backend_url,
            model="mock",
            n_requests=20,
            n_tenants=4,
            seed=7,
            concurrency=2,
            max_tokens=8,
            out_dir=tmp_path,
            modes=["baseline", "balanced", "protected_lane"],
        )

    assert "summary" in result
    names = [r["mode"] for r in result["summary"]]
    assert names == ["baseline", "balanced", "protected_lane"]
    # Every mode should have completed at least some requests.
    for r in result["summary"]:
        assert r["completed"] > 0
    # JSON file written
    assert result["json"].endswith("demo_results.json")

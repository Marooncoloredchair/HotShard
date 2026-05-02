"""Use guarded-hotshard in front of the OpenAI API itself.

The same `wrap()` works whether your backend is OpenAI, vLLM, Ollama, or
anything else that speaks the OpenAI API. The scheduler caps concurrency
locally and prioritizes premium tenants - so even when you're hitting
OpenAI's hosted API, your premium users get to the front of *your*
client-side queue first.

Run:
    OPENAI_API_KEY=sk-... python examples/openai_passthrough.py
"""

import asyncio
import time

from openai import OpenAI

from guarded_hotshard import wrap


def workload():
    """Mix of premium + bulk traffic."""
    return [
        {"user": "acme",   "msg": "Premium request 1"},
        {"user": "acme",   "msg": "Premium request 2"},
        {"user": "bulk-a", "msg": "Bulk request A1"},
        {"user": "bulk-a", "msg": "Bulk request A2"},
        {"user": "bulk-a", "msg": "Bulk request A3"},
        {"user": "acme",   "msg": "Premium request 3"},
        {"user": "bulk-b", "msg": "Bulk request B1"},
        {"user": "bulk-b", "msg": "Bulk request B2"},
    ]


def main() -> None:
    client = wrap(
        OpenAI(),
        mode="protected_lane",
        critical_users={"acme"},
        concurrency=2,   # keep low so the priority queue actually matters
    )

    for req in workload():
        t0 = time.time()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": req["msg"]}],
            user=req["user"],
            max_tokens=16,
        )
        print(f"[{req['user']:>6}]  {time.time() - t0:5.2f}s  -> {resp.choices[0].message.content[:40]}")


if __name__ == "__main__":
    main()

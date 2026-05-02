# Run guarded-hotshard in front of vLLM

vLLM speaks the OpenAI HTTP API on `/v1/*`, so you front it with `ghs serve`
without changing vLLM itself.

## 1. Start vLLM

```bash
pip install vllm

vllm serve Qwen/Qwen2.5-3B-Instruct \
    --port 8001 \
    --max-model-len 4096 \
    --dtype auto
```

If vLLM is already bound to `8000`, pick any free port for the backend and
keep `8000` for the proxy (or the reverse).

## 2. Start the proxy in front of it

```bash
pip install 'guarded-hotshard[server]'

ghs serve \
    --backend http://127.0.0.1:8001 \
    --port 8000 \
    --mode protected_lane \
    --critical-users acme-prod,bigco-tier1 \
    --storm-users loadtest-bot,noisy-tenant \
    --concurrency 8
```

- **`--backend`**: vLLM’s HTTP root (no trailing `/v1`; the proxy adds paths).
- **`--critical-users`**: OpenAI `user=` values that get priority + TMR-style
  redundancy when the mode enables it.
- **`--storm-users`**: tenants counted as storm-like in metrics (same as header
  `X-GHS-Storm: 1` on a request).

Smoke-check the stack:

```bash
curl -sS http://127.0.0.1:8000/healthz
curl -sS http://127.0.0.1:8000/metrics | findstr /I ghs_
```

## 3. Point your client at the proxy (8000), not vLLM (8001)

### Synchronous `OpenAI`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="-")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "hi"}],
    user="acme-prod",  # tenant id for scheduling
)
```

### Async `AsyncOpenAI`

```python
from openai import AsyncOpenAI
from guarded_hotshard import wrap_async

client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="-")
client = wrap_async(client, mode="protected_lane", critical_users={"acme-prod"})

resp = await client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "hi"}],
    user="acme-prod",
)
```

`user="…"` is what maps requests to tenants; `--critical-users` must use the
same strings.

## 4. Metrics while you load-test

`GET http://127.0.0.1:8000/metrics` returns Prometheus text format. Useful
series names include:

- `ghs_scheduler_queue_depth`, `ghs_scheduler_in_flight`
- `ghs_tenant_wall_seconds_p99` (label `tenant`)
- `ghs_backend_http_requests_total` (labels `role` = `primary` or `tmr`)
- `ghs_tmr_parallel_launches_total`, `ghs_protected_lane_tmr_activations_total`
- `ghs_storm_like_requests_total`

Wire these into Grafana or `promtool check metrics` while replaying traffic.

## 5. Demo: compare modes and open the HTML report

Point the demo at vLLM’s port (backend), not the proxy:

```bash
ghs demo \
    --backend http://127.0.0.1:8001 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --requests 150 \
    --concurrency 8 \
    --out demo_results
```

Outputs:

- `demo_results/demo_results.json` — full numbers
- `demo_results/report.html` — mode table, Pareto chart, per-tenant p99,
  cost/fidelity proxies (open in a browser)

The demo exercises the **scheduler + real backend** directly (no proxy hop).
Use it to pick a mode; use `ghs serve` in production to enforce that mode on
real client traffic.

# Run guarded-hotshard in front of vLLM

vLLM speaks the OpenAI API natively, so this is a one-liner.

## 1. Start vLLM

```bash
pip install vllm

vllm serve Qwen/Qwen2.5-3B-Instruct \
    --port 8001 \
    --max-model-len 4096 \
    --dtype auto
```

## 2. Start the proxy in front of it

```bash
pip install 'guarded-hotshard[server]'

ghs serve \
    --backend http://localhost:8001 \
    --port 8000 \
    --mode protected_lane \
    --critical-users acme-prod,bigco-tier1 \
    --concurrency 8
```

## 3. Point your client at the proxy (port 8000), not vLLM (port 8001)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "hi"}],
    user="acme-prod",   # this is your tenant id
)
```

`user="acme-prod"` is what makes the magic work: the proxy maps the OpenAI
`user` field to a tenant id and applies priority + TMR for traffic from
your `--critical-users` list.

## 4. Run the demo to compare modes head-to-head

```bash
ghs demo \
    --backend http://localhost:8001 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --requests 150 \
    --concurrency 8 \
    --out demo_results
```

This runs all six modes, prints a per-tenant p99 table and a Pareto chart,
and writes `demo_results/demo_results.json` for further analysis.

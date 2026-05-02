# Run guarded-hotshard in front of Ollama

Ollama 0.1.30+ exposes an OpenAI-compatible endpoint at `/v1`.

## 1. Pull a model and start Ollama

```bash
ollama pull qwen2.5:3b
ollama serve   # listens on :11434
```

## 2. Start the proxy

```bash
pip install 'guarded-hotshard[server]'

ghs serve \
    --backend http://localhost:11434 \
    --port 8000 \
    --mode balanced \
    --concurrency 4
```

(Ollama runs one request at a time on local hardware, but `concurrency=4`
gives the scheduler enough room to reorder a small queue.)

## 3. Use it like any OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")

resp = client.chat.completions.create(
    model="qwen2.5:3b",
    messages=[{"role": "user", "content": "hello"}],
    user="my-app-prod",
)
print(resp.choices[0].message.content)
```

## 4. Quick benchmark

```bash
ghs demo \
    --backend http://localhost:11434 \
    --model qwen2.5:3b \
    --requests 30 \
    --concurrency 2 \
    --max-tokens 16
```

(Keep `requests` and `concurrency` modest on local hardware - Ollama is
much slower than vLLM on a real GPU.)

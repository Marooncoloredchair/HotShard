# Contributing

Thanks for your interest in `guarded-hotshard`. This is a small, focused
project; PRs that keep it small and focused are very welcome.

## Quickstart

```bash
git clone https://github.com/Marooncoloredchair/HotShard.git
cd HotShard
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[all,dev]'
pytest -v
ruff check src tests
```

## What's in scope

- Bug fixes and tighter test coverage in `scheduler.py`, `layers.py`,
  `modes.py`.
- Better benchmarks (different models, larger workloads, real clusters).
- New OpenAI-compatible backend examples.
- Performance improvements that don't add dependencies.

## What's out of scope right now

- Vendor-specific integrations (e.g. directly hooking into vLLM's
  PagedAttention scheduler). These belong in a separate package.
- New scheduling laws / academic ideas without an experiment that
  validates them.
- Yet-another-config-format. We have six modes; that's enough.

## Mode contributions

If you want to add a new mode, please include:

1. A clear use case ("for X kind of workload, why").
2. A benchmark vs at least `baseline`, `balanced`, and `protected_lane`
   showing where it wins.
3. Tests in `tests/test_modes.py`.

## Running the demo

```bash
# Smoke test (mock backend, in CI):
pytest tests/test_demo_with_mock_backend.py -v

# Real backend (requires your own LLM server):
ghs demo --backend http://localhost:8001 --model qwen2.5-3b-instruct \
        --requests 60 --concurrency 4
```

## Style

- Format: ruff defaults.
- No new dependencies without a discussion.
- Keep type hints up to date.
- Prefer many small functions over few big ones.

## Releases

Maintainers cut releases by tagging `vX.Y.Z` and letting CI publish.

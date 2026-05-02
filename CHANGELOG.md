# Changelog

All notable changes to `guarded-hotshard` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-01

Initial public release.

### Added
- Six built-in scheduling modes: `baseline`, `eco`, `balanced`, `strict`,
  `critical`, `protected_lane`.
- `GuardedScheduler` core: in-process priority queue and async streaming
  dispatcher.
- `wrap()` for one-line OpenAI-compatible client wrapping.
- FastAPI proxy (`ghs serve`) that fronts any OpenAI-compatible backend
  (vLLM, Ollama, llama.cpp, OpenAI, Together, etc.) and adds
  tenant-aware priority scheduling and bounded redundancy.
- `ghs demo` CLI for end-to-end validation against any backend.
- Real-GPU benchmark on Colab Pro A100 + Qwen2.5-3B-Instruct: 60-70% T0
  p99 reduction at <5% extra cost across 3 seeds.

### Limitations
- Single-host scheduling. Multi-replica scheduling is on the roadmap.
- Output-distribution drift (`A_KLDrift`) requires a backend that exposes
  per-token logits; disabled in proxy mode by default.
- Lead-time predictor (`F_LeadTime`) is offline-trained today.

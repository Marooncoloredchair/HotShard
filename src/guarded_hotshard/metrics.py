"""Latency / fairness / cost metrics. Pure-Python, NumPy-only."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np


def percentile(values: Iterable[float], q: float) -> float:
    """Robust to empty input. q in [0, 1]."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


def jain_fairness(values: Iterable[float]) -> float:
    """Jain's fairness index. 1.0 = perfect equality, 1/n = worst."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 1.0
    s = arr.sum()
    if s == 0:
        return 1.0
    return float(s * s / (arr.size * np.sum(arr * arr)))


def per_tenant_p99(
    per_request: list[dict[str, Any]],
    latency_key: str = "wall_latency",
) -> dict[Hashable, float]:
    by_tenant: dict[Hashable, list[float]] = defaultdict(list)
    for r in per_request:
        by_tenant[r["tenant"]].append(float(r[latency_key]))
    return {t: percentile(v, 0.99) for t, v in by_tenant.items()}


def output_fidelity(
    baseline: dict[Hashable, str],
    candidate: dict[Hashable, str],
) -> float:
    """Token-overlap similarity vs baseline. 1.0 = identical, 0.0 = disjoint.

    We deliberately do NOT call this 'accuracy'. Without ground-truth answers
    we can only measure 'did we preserve baseline behavior?'.
    """
    if not baseline or not candidate:
        return 1.0
    scores = []
    for k, ref in baseline.items():
        if k not in candidate:
            continue
        scores.append(_token_overlap(ref, candidate[k]))
    if not scores:
        return 1.0
    return float(np.mean(scores))


def _token_overlap(a: str, b: str) -> float:
    ta = set((a or "").split())
    tb = set((b or "").split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def cost_per_million_tokens(
    total_wall_seconds: float,
    total_tokens: int,
    hourly_cost_usd: float = 4.0,
) -> float:
    """Modeled cost. Useful for relative comparison; absolute value depends on `hourly_cost_usd`."""
    if total_tokens <= 0:
        return 0.0
    cost_per_sec = hourly_cost_usd / 3600.0
    cost = total_wall_seconds * cost_per_sec
    return cost / (total_tokens / 1_000_000)


def summary_row(
    *,
    mode: str,
    completed: int,
    evicted: int,
    tmr: int,
    per_request: list[dict[str, Any]],
    total_wall_seconds: float,
    total_tokens: int,
    fidelity: float | None,
    hourly_cost_usd: float = 4.0,
    latency_key: str = "wall_latency",
) -> dict[str, Any]:
    """Build a single row of the summary table. Stable schema."""
    lats = [r[latency_key] for r in per_request]
    p50 = percentile(lats, 0.50)
    p99 = percentile(lats, 0.99)
    fairness = jain_fairness(lats)
    return {
        "mode": mode,
        "completed": completed,
        "evicted": evicted,
        "tmr": tmr,
        "p50_lat": round(p50, 3),
        "p99_lat": round(p99, 3),
        "tok/s": round(total_tokens / max(total_wall_seconds, 1e-6), 1),
        "fidelity": fidelity,
        "fairness": round(fairness, 3),
        "$/M_tok": round(
            cost_per_million_tokens(total_wall_seconds, total_tokens, hourly_cost_usd), 3
        ),
    }

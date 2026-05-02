"""Pluggable layers used by the scheduler.

Four small primitives the scheduler composes per-request:

    G - hot-tenant detection        (which tenants are sustained-load hot?)
    A - output-distribution drift   (does output diverge from reference?)
    F - lead-time predictor         (is an SLA breach impending?)
    D - budgeted re-execution       (do we pay for redundancy this tick?)

The layer-level API stays narrow on purpose. The composition choices live
in modes.py, where layers are wired into named operating points.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# G: hot-tenant promotion via hysteresis
# ---------------------------------------------------------------------------
class G_Hysteresis:
    """A tenant is marked hot only after `k` consecutive ticks above threshold.

    Hysteresis is the cheapest defense against thrash: a single noisy tick
    can't elevate a tenant, and a single quiet tick can't demote one.
    """

    def __init__(self, threshold: float = 0.6, k: int = 3):
        self.threshold = float(threshold)
        self.k = int(k)
        self.streak: dict[int, int] = defaultdict(int)
        self.hot: set[int] = set()

    def on_tick(self, tenant_id: int, load: float) -> bool:
        if load > self.threshold:
            self.streak[tenant_id] += 1
            if self.streak[tenant_id] >= self.k:
                self.hot.add(tenant_id)
        else:
            self.streak[tenant_id] = max(0, self.streak[tenant_id] - 1)
            if self.streak[tenant_id] == 0:
                self.hot.discard(tenant_id)
        return tenant_id in self.hot

    def is_hot(self, tenant_id: int) -> bool:
        return tenant_id in self.hot


# ---------------------------------------------------------------------------
# A: output-distribution drift detector (KL divergence)
# ---------------------------------------------------------------------------
class A_KLDrift:
    """KL divergence of recent output token distributions vs a running reference.

    Requires per-token logits. In proxy mode (where we only see decoded text
    coming back from the backend) this layer is bypassed; in tightly-coupled
    deployments where we have the model's first-token logits, it runs.
    """

    def __init__(self, threshold: float = 0.30, window: int = 32):
        self.threshold = float(threshold)
        self.window = int(window)
        self.ref_history: deque[np.ndarray] = deque(maxlen=window)

    def update_reference(self, probs: np.ndarray | None) -> None:
        if probs is None:
            return
        self.ref_history.append(np.asarray(probs, dtype=np.float64))

    def check(self, probs: np.ndarray | None) -> tuple[bool, float]:
        if probs is None or len(self.ref_history) < 4:
            return False, 0.0
        p = np.asarray(probs, dtype=np.float64)
        ref = np.mean(np.stack(self.ref_history, axis=0), axis=0)
        eps = 1e-9
        kl = float(np.sum(p * (np.log(p + eps) - np.log(ref + eps))))
        return kl > self.threshold, kl


# ---------------------------------------------------------------------------
# F: lead-time predictor for SLA breach
# ---------------------------------------------------------------------------
class F_LeadTime:
    """Tiny logistic-regression breach predictor.

    Trained online (or warmed up from a baseline trace). Optimised for
    *lead time*, not classification AUC. Calling `predict()` before `fit()`
    just returns 0.0 - the system gracefully degrades to D-only behavior.
    """

    def __init__(self, target_latency: float = 1.0, threshold: float = 0.30):
        self.target_latency = float(target_latency)
        self.threshold = float(threshold)
        self.X: list[list[float]] = []
        self.y: list[int] = []
        self.model = None  # sklearn LogisticRegression once fit

    @staticmethod
    def features(
        queue_len: int,
        batch_load: float,
        hot_count: int,
        tenant: int,
        recent_p99: float,
    ) -> list[float]:
        return [
            float(queue_len),
            float(batch_load),
            float(hot_count),
            float(tenant),
            float(recent_p99),
        ]

    def fit(self) -> bool:
        if len(self.X) < 16 or len(set(self.y)) < 2:
            return False
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: WPS433
        except ImportError:
            return False
        try:
            self.model = LogisticRegression(max_iter=200, class_weight="balanced")
            self.model.fit(self.X, self.y)
            return True
        except Exception:  # pragma: no cover  (sklearn convergence warnings, etc.)
            return False

    def predict(self, features: Iterable[float]) -> float:
        if self.model is None:
            return 0.0
        try:
            return float(self.model.predict_proba([list(features)])[0, 1])
        except Exception:  # pragma: no cover
            return 0.0


# ---------------------------------------------------------------------------
# D: budgeted re-execution
# ---------------------------------------------------------------------------
@dataclass
class D_Budgeted:
    """Bounded protection: pin requests for re-execution, but only `budget` per window.

    `fallback=True` means "re-execute even when the budget is exhausted"
    (unbounded protection). `fallback=False` means "accept the SLA breach"
    once the budget is spent, which is the cheaper and usually-correct move
    under cost pressure.
    """

    budget_per_window: int = 5
    window_ticks: int = 50
    fallback: bool = False
    risk_threshold: float = 0.5

    used: int = field(default=0, init=False)
    tick: int = field(default=0, init=False)

    @property
    def budget(self) -> int:
        return self.budget_per_window

    def reset(self) -> None:
        self.used = 0
        self.tick = 0

    def should_pin(self, F_risk: float, is_critical: bool) -> bool:
        self.tick += 1
        if self.tick >= self.window_ticks:
            self.tick = 0
            self.used = 0
        trigger = is_critical or F_risk >= self.risk_threshold
        if not trigger:
            return False
        if self.used < self.budget_per_window:
            self.used += 1
            return True
        return self.fallback

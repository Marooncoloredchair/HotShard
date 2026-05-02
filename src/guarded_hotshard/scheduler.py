"""Core scheduler. Pure-Python, no model dependencies.

Two flavours of operation:

1. Offline / batched (`schedule_batch`): take a list of pending requests,
   score them, pick which to admit, mark which to re-execute, and return
   the order the caller should run them in.

2. Online / streaming (`score`, `dispatch_next`, `complete`): a priority
   heap that the proxy or `wrap()` pulls from whenever a backend slot
   frees up.

The priority formula is intentionally simple and battle-tested. The
non-obvious behavior is exercised by `tests/test_priority_formula.py`;
treat that suite as the contract.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from typing import Any

from guarded_hotshard.modes import Mode


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------
@dataclass
class ScoredRequest:
    """A request after the scheduler has scored it.

    The original `request` payload is opaque to the scheduler (it's whatever
    the caller passed in). Everything the scheduler decides lives in the
    other fields.
    """

    request: Any
    request_id: Hashable
    tenant: Hashable
    is_critical: bool
    arrival_time: float
    priority: float
    pinned: bool        # D-Budgeted decided to re-execute this one
    hot: bool           # G-Hysteresis says the tenant is hot right now
    F_risk: float       # F-LeadTime's predicted breach probability
    tmr: bool = False   # post-admission decision: should we re-execute?

    # Internal heap ordering: highest priority first; tie-break on arrival.
    # A monotonically increasing seq prevents heapq from ever comparing
    # the request payloads themselves.
    _seq: int = field(default=0, repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_tenant(tenant: Hashable, critical_tenants: set[Hashable] | None) -> int:
    """Map an arbitrary tenant id to the integer the layers expect.

    G_Hysteresis stores streak counts keyed by the value passed in - it
    works fine with strings, but it's nicer to keep things consistent.
    The `is_critical` decision comes from `critical_tenants`, not from a
    magic tenant=0 check.
    """
    if isinstance(tenant, int):
        return tenant
    return hash(tenant) & 0xFFFFFFFF  # 32-bit, keeps streak dict happy


def score_one(
    *,
    mode: Mode,
    request: Any,
    request_id: Hashable,
    tenant: Hashable,
    is_critical: bool,
    arrival_time: float,
    queue_len: int,
    batch_load: float,
    recent_p99: float,
    hot_count: int,
) -> ScoredRequest:
    """Run one request through the G/F/D layers and compute its priority."""
    is_hot = mode.G.on_tick(_normalize_tenant(tenant, None), 1.0 if is_critical else 0.5)

    F_risk = 0.0
    if mode.use_F:
        feats = mode.F.features(queue_len, batch_load, hot_count, _normalize_tenant(tenant, None), recent_p99)
        F_risk = mode.F.predict(feats)

    is_pinned = mode.D.should_pin(F_risk, is_critical)

    if mode.use_priority:
        # Hot bonus only applies to non-critical traffic; the regression
        # test in tests/test_priority_formula.py locks this in.
        hot_bonus = 0.0 if is_critical else (10.0 * float(is_hot))
        priority = (
            100.0 * float(is_critical)
            + 50.0 * float(is_pinned)
            + hot_bonus
            - 0.1 * float(arrival_time)
        )
    else:
        priority = -float(arrival_time)

    return ScoredRequest(
        request=request,
        request_id=request_id,
        tenant=tenant,
        is_critical=is_critical,
        arrival_time=arrival_time,
        priority=priority,
        pinned=is_pinned,
        hot=is_hot,
        F_risk=F_risk,
    )


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------
class GuardedScheduler:
    """Stateful scheduler usable in both batch and streaming modes."""

    def __init__(
        self,
        mode: Mode,
        critical_tenants: Iterable[Hashable] | None = None,
        recent_p99_window: int = 32,
    ):
        self.mode = mode
        self.critical_tenants: set[Hashable] = set(critical_tenants or [])
        self._heap: list[tuple[float, int, ScoredRequest]] = []
        self._counter = itertools.count()
        self._in_flight = 0
        self._recent_lat: list[float] = []
        self._recent_p99_window = recent_p99_window

    # --- helpers --------------------------------------------------------
    def _is_critical(self, tenant: Hashable, override: bool | None) -> bool:
        if override is not None:
            return bool(override)
        if self.mode.tmr_target_tenants is not None:
            # protected_lane: tenants in the lane are the criticals
            return tenant in self.mode.tmr_target_tenants
        return tenant in self.critical_tenants

    def _recent_p99(self) -> float:
        if not self._recent_lat:
            return 0.0
        sorted_lat = sorted(self._recent_lat)
        idx = max(0, int(len(sorted_lat) * 0.99) - 1)
        return sorted_lat[idx]

    def hot_count(self) -> int:
        return len(self.mode.G.hot)

    def in_flight(self) -> int:
        return self._in_flight

    def queue_depth(self) -> int:
        return len(self._heap)

    # --- streaming API --------------------------------------------------
    def score(
        self,
        *,
        request: Any,
        request_id: Hashable,
        tenant: Hashable,
        arrival_time: float,
        is_critical: bool | None = None,
    ) -> ScoredRequest:
        crit = self._is_critical(tenant, is_critical)
        scored = score_one(
            mode=self.mode,
            request=request,
            request_id=request_id,
            tenant=tenant,
            is_critical=crit,
            arrival_time=arrival_time,
            queue_len=len(self._heap) + self._in_flight,
            batch_load=min(1.0, self._in_flight / 8.0),
            recent_p99=self._recent_p99(),
            hot_count=self.hot_count(),
        )
        # TMR decision in streaming mode: pin + (lane match if lane mode).
        # tmr_frac doesn't make sense per-request online, so we use the
        # D-pin signal directly (which is already budgeted).
        if scored.pinned:
            in_lane = (
                self.mode.tmr_target_tenants is None
                or tenant in self.mode.tmr_target_tenants
            )
            scored.tmr = in_lane and self.mode.tmr_frac > 0.0
        return scored

    def enqueue(self, scored: ScoredRequest) -> None:
        seq = next(self._counter)
        scored._seq = seq
        # heapq is a min-heap; we want highest priority first, so negate.
        heapq.heappush(self._heap, (-scored.priority, seq, scored))

    def dispatch_next(self) -> ScoredRequest | None:
        """Pop the highest-priority queued request. Returns None if empty."""
        if not self._heap:
            return None
        _, _, scored = heapq.heappop(self._heap)
        self._in_flight += 1
        return scored

    def complete(self, scored: ScoredRequest, latency: float) -> None:
        """Tell the scheduler a dispatched request finished. Updates online stats."""
        self._in_flight = max(0, self._in_flight - 1)
        self._recent_lat.append(float(latency))
        if len(self._recent_lat) > self._recent_p99_window:
            self._recent_lat = self._recent_lat[-self._recent_p99_window :]

    # --- batch API ------------------------------------------------------
    def schedule_batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        """One-shot scheduling for an offline workload.

        Each request dict must have:
            id, tenant, arrival_time
        Optional:
            is_critical (bool)  - else taken from `critical_tenants` / lane
            anything else (passed through opaquely)

        Returns a dict with `admitted`, `evicted`, `tmr_set` (set of ids),
        and the full `scored` list in priority order.
        """
        scored: list[ScoredRequest] = []
        n = len(requests)
        for i, req in enumerate(requests):
            crit = self._is_critical(req["tenant"], req.get("is_critical"))
            sr = score_one(
                mode=self.mode,
                request=req,
                request_id=req["id"],
                tenant=req["tenant"],
                is_critical=crit,
                arrival_time=float(req.get("arrival_time", i)),
                queue_len=n - i,
                batch_load=i / max(1, n),
                recent_p99=0.0,
                hot_count=int(crit),
            )
            scored.append(sr)

        if self.mode.use_priority:
            scored.sort(key=lambda s: -s.priority)

        # Eviction: drop the bottom `eviction_frac`.
        n_evict = int(self.mode.eviction_frac * len(scored))
        admitted = scored[: len(scored) - n_evict] if n_evict else scored
        evicted = scored[len(scored) - n_evict :] if n_evict else []

        # TMR set:
        n_tmr = int(self.mode.tmr_frac * len(admitted))
        if self.mode.tmr_target_tenants is not None:
            pool = [s for s in admitted if s.tenant in self.mode.tmr_target_tenants]
            tmr_candidates = pool[:n_tmr]
        else:
            tmr_candidates = admitted[:n_tmr]
        tmr_set: set[Hashable] = {s.request_id for s in tmr_candidates}
        for s in admitted:
            s.tmr = s.request_id in tmr_set

        return {
            "scored": scored,
            "admitted": admitted,
            "evicted": evicted,
            "tmr_set": tmr_set,
        }

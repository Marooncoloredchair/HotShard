"""Smoke tests for the scheduler's batch + streaming APIs."""

from __future__ import annotations

from guarded_hotshard import GuardedScheduler, make_mode


def _zipf_workload(n: int = 60, seed: int = 0) -> list[dict]:
    import numpy as np

    rng = np.random.default_rng(seed)
    weights = 1.0 / (np.arange(1, 6) ** 1.2)
    weights /= weights.sum()
    out = []
    for i in range(n):
        t = int(rng.choice(5, p=weights))
        out.append(
            {
                "id": i,
                "tenant": t,
                "is_critical": t == 0,
                "arrival_time": i * 0.05,
            }
        )
    return out


def test_baseline_does_not_reorder():
    sched = GuardedScheduler(make_mode("baseline"))
    wl = _zipf_workload(20, seed=0)
    plan = sched.schedule_batch(wl)
    # Baseline disables priority, so admit order == input order.
    ids = [s.request_id for s in plan["admitted"]]
    assert ids == [r["id"] for r in wl]
    assert plan["evicted"] == []
    assert plan["tmr_set"] == set()


def test_eco_evicts_15_percent():
    sched = GuardedScheduler(make_mode("eco"))
    wl = _zipf_workload(60, seed=0)
    plan = sched.schedule_batch(wl)
    assert len(plan["evicted"]) == 9  # 0.15 * 60
    assert len(plan["admitted"]) == 51
    assert len(plan["tmr_set"]) == 0


def test_balanced_pins_some_critical_for_tmr():
    sched = GuardedScheduler(make_mode("balanced"))
    wl = _zipf_workload(60, seed=0)
    plan = sched.schedule_batch(wl)
    n_admit = len(plan["admitted"])
    assert n_admit == 57  # 0.05 eviction
    # tmr_frac=0.05 of admitted, rounded down.
    assert len(plan["tmr_set"]) == int(0.05 * n_admit)
    # All TMR'd requests should be critical (T0) in this workload, because
    # admitted is sorted by priority and crits sort first.
    for s in plan["admitted"]:
        if s.request_id in plan["tmr_set"]:
            assert s.tenant == 0


def test_protected_lane_only_tmrs_target_tenant():
    sched = GuardedScheduler(make_mode("protected_lane"))
    wl = _zipf_workload(60, seed=0)
    plan = sched.schedule_batch(wl)
    for s in plan["admitted"]:
        if s.tmr:
            assert s.tenant == 0


def test_streaming_dispatches_highest_priority_first():
    # Tenant 0 is the premium tenant; we tell the scheduler at construction.
    sched = GuardedScheduler(make_mode("balanced"), critical_tenants={0})
    for i in range(10):
        sr = sched.score(
            request={"id": i},
            request_id=i,
            tenant=0 if i in (3, 7) else 1,
            arrival_time=i * 0.05,
        )
        sched.enqueue(sr)
    order = []
    while sched.queue_depth() > 0:
        sr = sched.dispatch_next()
        order.append(sr.request_id)
        sched.complete(sr, 0.1)
    # The two critical T0 requests (ids 3 and 7) must come out first.
    assert order[0] in (3, 7)
    assert order[1] in (3, 7)


def test_streaming_explicit_is_critical_overrides_set():
    """Even without a `critical_tenants` set, an explicit `is_critical=True` wins."""
    sched = GuardedScheduler(make_mode("balanced"))
    sr1 = sched.score(request={}, request_id=1, tenant="bulk", arrival_time=0.0)
    sr2 = sched.score(
        request={}, request_id=2, tenant="bulk", arrival_time=0.5, is_critical=True
    )
    sched.enqueue(sr1)
    sched.enqueue(sr2)
    first = sched.dispatch_next()
    assert first.request_id == 2

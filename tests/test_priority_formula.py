"""Regression tests for the priority formula.

These tests pin down two invariants that the priority calculation must
preserve, regardless of how the layer parameters are tuned:

  1. Within the critical bucket, requests sort by arrival time only.
     Hot-tenant detection state must not create sub-tiers among critical
     traffic (otherwise the first few critical requests in a trace are
     deprioritized while the hot-tenant detector warms up).

  2. A single critical request always outranks any non-critical traffic,
     even when the non-critical tenant is hot.

If you change `score_one()` or any priority weights, these tests must
continue to pass.
"""

from __future__ import annotations

from guarded_hotshard import GuardedScheduler, make_mode


def _make_workload() -> list[dict]:
    workload: list[dict] = []
    # 10 critical requests at t=0..0.45
    for i in range(10):
        workload.append(
            {"id": i, "tenant": 0, "is_critical": True, "arrival_time": i * 0.05}
        )
    # 5 non-critical requests later
    for i in range(5):
        workload.append(
            {
                "id": 100 + i,
                "tenant": 1,
                "is_critical": False,
                "arrival_time": 1.0 + i * 0.05,
            }
        )
    return workload


def test_protected_lane_keeps_earliest_critical_first():
    sched = GuardedScheduler(make_mode("protected_lane"))
    plan = sched.schedule_batch(_make_workload())

    # Find the order of critical (T0) requests post-sort.
    crit_order = [s.request_id for s in plan["admitted"] if s.tenant == 0]
    # The earliest-arriving critical request must be first in the admit order.
    assert crit_order[0] == 0, (
        f"expected request 0 to lead the critical lane, got {crit_order}"
    )
    # And the order should be strictly ascending by id (= ascending by arrival).
    assert crit_order == sorted(crit_order), f"critical lane out of order: {crit_order}"


def test_balanced_keeps_earliest_critical_first():
    sched = GuardedScheduler(make_mode("balanced"))
    plan = sched.schedule_batch(_make_workload())
    crit_order = [s.request_id for s in plan["admitted"] if s.tenant == 0]
    assert crit_order[0] == 0
    assert crit_order == sorted(crit_order)


def test_strict_keeps_earliest_critical_first():
    sched = GuardedScheduler(make_mode("strict"))
    plan = sched.schedule_batch(_make_workload())
    crit_order = [s.request_id for s in plan["admitted"] if s.tenant == 0]
    assert crit_order[0] == 0
    assert crit_order == sorted(crit_order)


def test_critical_mode_keeps_earliest_critical_first():
    sched = GuardedScheduler(make_mode("critical"))
    plan = sched.schedule_batch(_make_workload())
    crit_order = [s.request_id for s in plan["admitted"] if s.tenant == 0]
    assert crit_order[0] == 0
    assert crit_order == sorted(crit_order)


def test_critical_priority_dominates_hot():
    """A non-hot critical must outrank a hot non-critical."""
    sched = GuardedScheduler(make_mode("balanced"))
    workload = [
        # First, send 5 non-critical T1 requests to make T1 go hot.
        {"id": 1, "tenant": 1, "is_critical": False, "arrival_time": 0.0},
        {"id": 2, "tenant": 1, "is_critical": False, "arrival_time": 0.1},
        {"id": 3, "tenant": 1, "is_critical": False, "arrival_time": 0.2},
        {"id": 4, "tenant": 1, "is_critical": False, "arrival_time": 0.3},
        {"id": 5, "tenant": 1, "is_critical": False, "arrival_time": 0.4},
        # Then a single T0 critical request after T1 has gone hot.
        {"id": 99, "tenant": 0, "is_critical": True, "arrival_time": 0.5},
    ]
    plan = sched.schedule_batch(workload)
    # Critical must come first regardless of T1's hotness.
    assert plan["admitted"][0].request_id == 99

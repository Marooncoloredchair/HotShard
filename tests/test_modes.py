"""Tests for the built-in mode catalogue."""

from __future__ import annotations

import pytest

from guarded_hotshard import MODES, Mode, make_mode


def test_public_modes_list():
    """The public modes list is what `ghs modes` shows and the demo runs."""
    expected = {"baseline", "eco", "balanced", "protected_lane", "critical"}
    assert set(MODES) == expected


def test_strict_still_constructable():
    """`strict` is no longer public-facing but stays callable for advanced users."""
    m = make_mode("strict")
    assert m.name == "strict"


@pytest.mark.parametrize("name", ["baseline", "eco", "balanced", "strict", "critical", "protected_lane"])
def test_make_mode_returns_fresh_instance(name: str):
    a = make_mode(name)
    b = make_mode(name)
    assert isinstance(a, Mode)
    assert a.name == name
    assert b.name == name
    # State should be independent.
    a.G.streak[0] = 99
    assert b.G.streak.get(0, 0) == 0


def test_baseline_disables_priority_and_protection():
    m = make_mode("baseline")
    assert m.use_priority is False
    assert m.eviction_frac == 0.0
    assert m.tmr_frac == 0.0
    assert m.use_F is False


def test_eco_evicts_but_does_not_tmr():
    m = make_mode("eco")
    assert m.eviction_frac > 0
    assert m.tmr_frac == 0.0


def test_critical_uses_unbounded_fallback():
    m = make_mode("critical")
    assert m.D.fallback is True
    assert m.tmr_frac >= 0.20


def test_protected_lane_targets_tenant_zero_by_default():
    m = make_mode("protected_lane")
    assert m.tmr_target_tenants == {0}
    assert m.tmr_frac > 0
    # Bounded protection: D.fallback should NOT be True for the buyer's mode.
    assert m.D.fallback is False


def test_protected_lane_accepts_custom_critical_tenants():
    m = make_mode("protected_lane", critical_tenants={"acme", "globex"})
    assert m.tmr_target_tenants == {"acme", "globex"}


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        make_mode("not-a-mode")

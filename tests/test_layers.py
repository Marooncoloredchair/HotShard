"""Tests for individual HCF layers."""

from __future__ import annotations

import numpy as np

from guarded_hotshard import A_KLDrift, D_Budgeted, F_LeadTime, G_Hysteresis

# G_Hysteresis -------------------------------------------------------------

def test_hysteresis_promotes_after_k_consecutive_ticks():
    g = G_Hysteresis(threshold=0.5, k=3)
    assert g.on_tick(1, 0.7) is False  # streak=1
    assert g.on_tick(1, 0.7) is False  # streak=2
    assert g.on_tick(1, 0.7) is True   # streak=3 -> hot
    assert g.is_hot(1) is True


def test_hysteresis_demotes_when_streak_falls_to_zero():
    g = G_Hysteresis(threshold=0.5, k=2)
    g.on_tick(1, 0.9)
    g.on_tick(1, 0.9)
    assert g.is_hot(1) is True
    g.on_tick(1, 0.0)
    g.on_tick(1, 0.0)
    assert g.is_hot(1) is False


def test_hysteresis_isolates_tenants():
    g = G_Hysteresis(threshold=0.5, k=2)
    g.on_tick(1, 0.9)
    g.on_tick(1, 0.9)
    assert g.is_hot(1) is True
    assert g.is_hot(2) is False


# A_KLDrift ----------------------------------------------------------------

def test_kldrift_returns_false_with_insufficient_history():
    a = A_KLDrift(threshold=0.1)
    a.update_reference(np.array([0.5, 0.5]))
    flagged, kl = a.check(np.array([0.9, 0.1]))
    assert flagged is False
    assert kl == 0.0


def test_kldrift_flags_when_distribution_shifts():
    a = A_KLDrift(threshold=0.1)
    for _ in range(5):
        a.update_reference(np.array([0.5, 0.5]))
    flagged, kl = a.check(np.array([0.95, 0.05]))
    assert kl > 0.1
    assert flagged is True


# F_LeadTime ---------------------------------------------------------------

def test_leadtime_predict_zero_without_fit():
    f = F_LeadTime()
    assert f.predict([0, 0, 0, 0, 0]) == 0.0


def test_leadtime_fit_returns_false_with_too_few_samples():
    f = F_LeadTime()
    for i in range(10):
        f.X.append([float(i), 0, 0, 0, 0])
        f.y.append(i % 2)
    assert f.fit() is False  # need >= 16


# D_Budgeted ---------------------------------------------------------------

def test_dbudgeted_pins_within_budget():
    d = D_Budgeted(budget_per_window=2, window_ticks=10, fallback=False, risk_threshold=0.5)
    assert d.should_pin(0.0, is_critical=True) is True   # 1
    assert d.should_pin(0.0, is_critical=True) is True   # 2
    assert d.should_pin(0.0, is_critical=True) is False  # exhausted, fallback=False


def test_dbudgeted_with_fallback_pins_beyond_budget():
    d = D_Budgeted(budget_per_window=1, window_ticks=10, fallback=True, risk_threshold=0.5)
    assert d.should_pin(0.0, is_critical=True) is True
    assert d.should_pin(0.0, is_critical=True) is True  # fallback


def test_dbudgeted_does_not_pin_below_threshold():
    d = D_Budgeted(budget_per_window=5, risk_threshold=0.5)
    assert d.should_pin(F_risk=0.1, is_critical=False) is False


def test_dbudgeted_resets_after_window():
    d = D_Budgeted(budget_per_window=1, window_ticks=3, fallback=False, risk_threshold=0.5)
    assert d.should_pin(0.0, is_critical=True) is True
    assert d.should_pin(0.0, is_critical=True) is False
    # Tick over the window.
    d.should_pin(0.0, is_critical=False)
    # Now budget is fresh.
    assert d.should_pin(0.0, is_critical=True) is True

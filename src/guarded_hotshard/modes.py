"""Named operating points for the scheduler.

A `Mode` bundles configured layers plus a few scheduler-level knobs
(eviction fraction, redundancy fraction, target tenants) into a single
profile. Six modes ship out of the box; for custom modes, instantiate a
`Mode` directly.

The exact thresholds and budgets ship as defaults that have been validated
on a real-GPU benchmark; see the README's Pareto frontier for the
trade-offs each mode hits.
"""

from __future__ import annotations

from dataclasses import dataclass

from guarded_hotshard.layers import (
    A_KLDrift,
    D_Budgeted,
    F_LeadTime,
    G_Hysteresis,
)


@dataclass
class Mode:
    name: str
    G: G_Hysteresis
    A: A_KLDrift
    F: F_LeadTime
    D: D_Budgeted
    use_priority: bool = True
    eviction_frac: float = 0.0
    tmr_frac: float = 0.0
    use_F: bool = True
    tmr_target_tenants: set[int] | None = None
    description: str = ""

    def reset(self) -> None:
        """Reset stateful layers between runs (useful in tests / repeated demos)."""
        self.G.streak.clear()
        self.G.hot.clear()
        self.A.ref_history.clear()
        self.D.reset()


def _baseline() -> Mode:
    return Mode(
        name="baseline",
        G=G_Hysteresis(threshold=99.0, k=99),
        A=A_KLDrift(threshold=99.0),
        F=F_LeadTime(threshold=99.0),
        D=D_Budgeted(budget_per_window=0, fallback=False),
        use_priority=False,
        eviction_frac=0.0,
        tmr_frac=0.0,
        use_F=False,
        description="No HCF. FIFO admission, no priority, no TMR. Reference output.",
    )


def _eco() -> Mode:
    return Mode(
        name="eco",
        G=G_Hysteresis(threshold=0.8, k=5),
        A=A_KLDrift(threshold=0.50),
        F=F_LeadTime(threshold=99.0),
        D=D_Budgeted(budget_per_window=0, fallback=False),
        use_priority=True,
        eviction_frac=0.15,
        tmr_frac=0.0,
        use_F=False,
        description="Aggressive eviction, no TMR. Lowest cost.",
    )


def _balanced() -> Mode:
    return Mode(
        name="balanced",
        G=G_Hysteresis(threshold=0.6, k=3),
        A=A_KLDrift(threshold=0.30),
        F=F_LeadTime(threshold=0.40),
        D=D_Budgeted(
            budget_per_window=4, window_ticks=20, fallback=False, risk_threshold=0.40
        ),
        use_priority=True,
        eviction_frac=0.05,
        tmr_frac=0.05,
        use_F=True,
        description="Hot-tenant promotion + light F + budgeted TMR for criticals.",
    )


def _strict() -> Mode:
    return Mode(
        name="strict",
        G=G_Hysteresis(threshold=0.5, k=2),
        A=A_KLDrift(threshold=0.25),
        F=F_LeadTime(threshold=0.30),
        D=D_Budgeted(
            budget_per_window=8, window_ticks=20, fallback=True, risk_threshold=0.30
        ),
        use_priority=True,
        eviction_frac=0.0,
        tmr_frac=0.10,
        use_F=True,
        description="Tighter F, larger TMR budget, unbounded fallback.",
    )


def _critical() -> Mode:
    return Mode(
        name="critical",
        G=G_Hysteresis(threshold=0.4, k=1),
        A=A_KLDrift(threshold=0.20),
        F=F_LeadTime(threshold=0.20),
        D=D_Budgeted(
            budget_per_window=20, window_ticks=20, fallback=True, risk_threshold=0.20
        ),
        use_priority=True,
        eviction_frac=0.0,
        tmr_frac=0.30,
        use_F=True,
        description="Heavy TMR, no eviction, full per-tenant priority.",
    )


def _protected_lane(critical_tenants: set[int] | None = None) -> Mode:
    """Balanced globally, but redundancy fires only for premium tenants.

    The default 'premium lane' is `{0}` (tenant 0). Pass `critical_tenants`
    to override - e.g. `make_mode("protected_lane", critical_tenants={"acme"})`.

    Defaults shipped here are conservative starting points. Workload-specific
    tuning (different threshold/budget/window combinations) is the usual
    way to push p99 reduction further on a given backend.
    """
    return Mode(
        name="protected_lane",
        G=G_Hysteresis(threshold=0.7, k=3),
        A=A_KLDrift(threshold=0.30),
        F=F_LeadTime(threshold=0.30),
        D=D_Budgeted(
            budget_per_window=10, window_ticks=20, fallback=False, risk_threshold=0.40
        ),
        use_priority=True,
        eviction_frac=0.05,
        tmr_frac=0.08,
        tmr_target_tenants=critical_tenants if critical_tenants is not None else {0},
        use_F=True,
        description="Balanced globally + premium-lane redundancy for premium tenants.",
    )


_FACTORIES = {
    "baseline": _baseline,
    "eco": _eco,
    "balanced": _balanced,
    "strict": _strict,
    "critical": _critical,
    "protected_lane": _protected_lane,
}


MODES: list[str] = ["baseline", "eco", "balanced", "protected_lane", "critical"]
"""Public mode catalogue, ordered cheapest-to-most-protective.

Other operating points (with more aggressive tuning) are available via
direct `Mode(...)` construction or commercial support; see CONTRIBUTING.md.
"""


def make_mode(name: str, **kwargs) -> Mode:
    """Build a fresh `Mode` by name.

    Each call returns a brand-new mode with fresh layer state, so it's safe
    to reuse the same name across runs. `kwargs` are forwarded to the
    factory; today only `protected_lane` accepts `critical_tenants`.
    """
    if name not in _FACTORIES:
        raise ValueError(
            f"Unknown mode {name!r}. Built-in modes: {MODES}. "
            "For custom modes, construct a Mode() directly."
        )
    return _FACTORIES[name](**kwargs)

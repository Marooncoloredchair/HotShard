"""guarded-hotshard: tenant-aware request scheduling for LLM inference.

Public surface kept small on purpose. Most users only ever import:

    from guarded_hotshard import wrap, GuardedScheduler, make_mode

See https://github.com/Marooncoloredchair/HotShard for docs and benchmarks.
"""

from guarded_hotshard._version import __version__
from guarded_hotshard.layers import (
    A_KLDrift,
    D_Budgeted,
    F_LeadTime,
    G_Hysteresis,
)
from guarded_hotshard.modes import MODES, Mode, make_mode
from guarded_hotshard.scheduler import GuardedScheduler, ScoredRequest
from guarded_hotshard.wrap import wrap

__all__ = [
    "__version__",
    "Mode",
    "MODES",
    "make_mode",
    "GuardedScheduler",
    "ScoredRequest",
    "wrap",
    "G_Hysteresis",
    "A_KLDrift",
    "F_LeadTime",
    "D_Budgeted",
]

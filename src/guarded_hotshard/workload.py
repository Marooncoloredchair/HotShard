"""Synthetic Zipf-skewed multi-tenant chat workload, ported from the Colab notebook.

Used by `ghs demo` and the offline tests. Real applications will plug in
their own request streams, but having a deterministic synthetic workload is
crucial for reproducible benchmarks and for cold demos where you don't yet
have access to the customer's traffic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SYSTEM_PROMPTS = [
    "You are a helpful AI assistant.",
    "You are a customer service agent for a software company.",
    "You are a code review assistant focused on Python.",
    "You are a creative writing partner.",
    "You are a math tutor for high school students.",
]

USER_TEMPLATES = [
    "Tell me about {topic} in two sentences.",
    "Explain {topic} like I'm new to it.",
    "What's the relationship between {topic} and {topic2}?",
    "Give me three quick facts about {topic}.",
    "Write a single short paragraph about {topic}.",
    "Summarize {topic} in one line.",
]

TOPICS = [
    "photosynthesis", "quantum mechanics", "the French Revolution",
    "machine learning", "the ocean", "ancient Rome", "DNA",
    "black holes", "the brain", "climate change", "the printing press",
    "Python decorators", "neural networks", "evolution", "tectonics",
]


def make_workload(
    n_requests: int = 60,
    n_tenants: int = 5,
    alpha: float = 1.2,
    seed: int = 42,
    adversarial: bool = False,
) -> list[dict[str, Any]]:
    """Generate a Zipf-skewed multi-tenant chat workload.

    If `adversarial` is True, a non-critical 'storm tenant' (id=n_tenants)
    floods 30% of requests during the middle 40% of the trace with cheap
    cache-pollution prompts. Used to verify HCF does NOT promote them
    despite their high volume (Law 7).
    """
    rng = np.random.default_rng(seed)
    weights = 1.0 / (np.arange(1, n_tenants + 1) ** alpha)
    weights /= weights.sum()
    storm_tenant = n_tenants
    storm_window = (int(n_requests * 0.3), int(n_requests * 0.7))

    out: list[dict[str, Any]] = []
    for i in range(n_requests):
        tenant = int(rng.choice(n_tenants, p=weights))
        if 25 <= i < 40 and rng.random() < 0.5:
            tenant = 1
        is_storm = False
        if adversarial and storm_window[0] <= i < storm_window[1] and rng.random() < 0.30:
            tenant = storm_tenant
            is_storm = True
        is_critical = (tenant == 0)
        system = SYSTEM_PROMPTS[tenant % len(SYSTEM_PROMPTS)]
        if is_storm:
            user = "ping"
        else:
            topic = str(rng.choice(TOPICS))
            topic2 = str(rng.choice(TOPICS))
            user = str(rng.choice(USER_TEMPLATES)).format(topic=topic, topic2=topic2)
        out.append(
            {
                "id": i,
                "tenant": tenant,
                "system": system,
                "user": user,
                "is_critical": is_critical,
                "is_storm": is_storm,
                "arrival_time": float(i) * 0.05,
            }
        )
    return out

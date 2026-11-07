"""Bootstrap confidence intervals, including paired differences between configurations."""

from __future__ import annotations

import math
import random


def mean(values: list[float]) -> float:
    """Mean of the finite values (NaN if there are none)."""
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean of ``values`` (NaNs dropped)."""
    data = [v for v in values if math.isfinite(v)]
    if not data:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(data)
    stats = sorted(sum(data[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return stats[int(0.025 * samples)], stats[int(0.975 * samples) - 1]


def paired_difference(a: list[float], b: list[float], samples: int, seed: int) -> tuple[float, float, float, int]:
    """Mean of (a - b) over items scored in both, with a paired 95% bootstrap interval.

    Returns ``(mean_difference, low, high, n_pairs)``.
    """
    pairs = [x - y for x, y in zip(a, b, strict=True) if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return float("nan"), float("nan"), float("nan"), 0
    low, high = bootstrap_ci(pairs, samples, seed)
    return sum(pairs) / len(pairs), low, high, len(pairs)


def verdict(low: float, high: float, name_a: str, name_b: str, higher_is_better: bool = True) -> str:
    """Plain-language reading of a paired interval."""
    if not (math.isfinite(low) and math.isfinite(high)) or low <= 0 <= high:
        return "no clear difference"
    a_higher = low > 0
    better = name_a if a_higher == higher_is_better else name_b
    return f"{better} better"

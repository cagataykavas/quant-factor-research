from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from math import floor, isfinite, sqrt

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True, slots=True)
class HACMeanInference:
    observations: int
    mean: float
    lag: int
    bandwidth_method: str
    naive_standard_error: float
    hac_standard_error: float
    long_run_variance: float
    standard_error_ratio: float | None
    t_statistic: float | None
    two_sided_p_value: float | None

    def as_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def automatic_newey_west_lag(observations: int) -> int:
    """Return the common floor(4 * (T / 100) ** (2 / 9)) bandwidth."""
    if observations < 3:
        raise ValueError("at least three observations are required")
    return min(observations - 1, floor(4 * (observations / 100) ** (2 / 9)))


def hac_mean_inference(
    values: Iterable[float],
    *,
    lag: int | None = None,
) -> HACMeanInference:
    """Estimate mean uncertainty with a Bartlett-kernel Newey-West variance.

    The p-value uses the asymptotic standard-normal reference distribution. It
    is diagnostic evidence for a prespecified IC series, not a data-snooping or
    multiple-testing correction.
    """
    sample = np.asarray(list(values), dtype=float)
    if sample.ndim != 1 or len(sample) < 3:
        raise ValueError("at least three one-dimensional observations are required")
    if not np.isfinite(sample).all():
        raise ValueError("observations must be finite")
    if np.any(np.abs(sample) > 1):
        raise ValueError("information coefficients must be in [-1, 1]")

    if lag is None:
        selected_lag = automatic_newey_west_lag(len(sample))
        method = "automatic"
    else:
        if isinstance(lag, bool) or not isinstance(lag, int):
            raise TypeError("lag must be an integer")
        if not 0 <= lag < len(sample):
            raise ValueError("lag must be between zero and observations minus one")
        selected_lag = lag
        method = "explicit"

    mean = float(np.mean(sample))
    centered = sample - mean
    observations = len(sample)
    naive_standard_error = float(np.std(sample, ddof=1) / sqrt(observations))
    long_run_variance = float(np.dot(centered, centered) / observations)
    for offset in range(1, selected_lag + 1):
        weight = 1 - offset / (selected_lag + 1)
        autocovariance = float(
            np.dot(centered[offset:], centered[:-offset]) / observations
        )
        long_run_variance += 2 * weight * autocovariance
    # Finite samples can produce a slightly negative estimate under strong
    # negative serial dependence. Zero is the variance boundary; inference is
    # reported as unavailable rather than manufacturing an infinite statistic.
    long_run_variance = max(0.0, long_run_variance)
    hac_standard_error = sqrt(long_run_variance / observations)
    ratio = (
        hac_standard_error / naive_standard_error if naive_standard_error > 0 else None
    )
    if hac_standard_error > 0:
        t_statistic = mean / hac_standard_error
        p_value = float(2 * norm.sf(abs(t_statistic)))
        if not isfinite(p_value):  # pragma: no cover - defensive SciPy boundary
            raise ValueError("HAC p-value is not finite")
    else:
        t_statistic = None
        p_value = None

    return HACMeanInference(
        observations=observations,
        mean=mean,
        lag=selected_lag,
        bandwidth_method=method,
        naive_standard_error=naive_standard_error,
        hac_standard_error=hac_standard_error,
        long_run_variance=long_run_variance,
        standard_error_ratio=ratio,
        t_statistic=t_statistic,
        two_sided_p_value=p_value,
    )

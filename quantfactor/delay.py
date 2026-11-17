from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from .core import information_coefficient

MAX_LAGS = 10
MAX_LAG = 63
MAX_COMPARISON_CELLS = 10_000_000


class InvalidDelayEvidence(ValueError):
    """Raised when signal-delay evidence cannot be evaluated safely."""


@dataclass(frozen=True)
class SignalDelayPolicy:
    lags: tuple[int, ...] = (0, 1, 2, 5)
    required_lag: int = 1
    minimum_assets_per_date: int = 5
    minimum_observations: int = 60
    minimum_reference_abs_ic: float = 0.01
    minimum_required_lag_retention: float = 0.5


@dataclass(frozen=True)
class SignalDelayPoint:
    lag: int
    observations: int
    mean_ic: float
    median_ic: float
    positive_ic_rate: float
    oriented_ic_retention: float | None


@dataclass(frozen=True)
class SignalDelayReport:
    schema_version: int
    accepted: bool
    reasons: tuple[str, ...]
    required_lag: int
    reference_mean_ic: float
    reference_direction: str
    required_lag_retention: float | None
    policy: SignalDelayPolicy
    points: tuple[SignalDelayPoint, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "reasons": list(self.reasons),
            "policy": asdict(self.policy),
            "points": [asdict(point) for point in self.points],
        }


def _validate_policy(policy: SignalDelayPolicy) -> None:
    lags = policy.lags
    if not isinstance(lags, tuple) or not 2 <= len(lags) <= MAX_LAGS:
        raise InvalidDelayEvidence(
            f"policy.lags must contain 2 through {MAX_LAGS} lags"
        )
    if any(isinstance(lag, bool) or not isinstance(lag, int) for lag in lags):
        raise InvalidDelayEvidence("policy.lags must contain integers")
    if tuple(sorted(set(lags))) != lags or lags[0] != 0 or lags[-1] > MAX_LAG:
        raise InvalidDelayEvidence(
            f"policy.lags must be unique, increasing, start at 0, and not exceed {MAX_LAG}"
        )
    if policy.required_lag not in lags or policy.required_lag == 0:
        raise InvalidDelayEvidence(
            "policy.required_lag must be a nonzero configured lag"
        )
    for field, lower in (("minimum_assets_per_date", 3), ("minimum_observations", 20)):
        value = getattr(policy, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise InvalidDelayEvidence(
                f"policy.{field} must be an integer of at least {lower}"
            )
    for field in ("minimum_reference_abs_ic", "minimum_required_lag_retention"):
        value = getattr(policy, field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidDelayEvidence(f"policy.{field} must be numeric")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise InvalidDelayEvidence(
                f"policy.{field} must be finite and between 0 and 1"
            )


def _validate_panel(
    signal_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    policy: SignalDelayPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal_panel.empty or forward_returns.empty:
        raise InvalidDelayEvidence("signal and forward-return panels must be non-empty")
    if signal_panel.index.has_duplicates or forward_returns.index.has_duplicates:
        raise InvalidDelayEvidence("panel date indexes must be unique")
    if signal_panel.columns.has_duplicates or forward_returns.columns.has_duplicates:
        raise InvalidDelayEvidence("panel asset columns must be unique")
    if not signal_panel.index.is_monotonic_increasing:
        raise InvalidDelayEvidence("signal dates must be increasing")
    if not signal_panel.index.equals(forward_returns.index):
        raise InvalidDelayEvidence("signal and forward-return dates must match exactly")
    if set(signal_panel.columns) != set(forward_returns.columns):
        raise InvalidDelayEvidence(
            "signal and forward-return assets must match exactly"
        )
    if len(signal_panel) <= policy.lags[-1]:
        raise InvalidDelayEvidence("not enough dates for the largest configured lag")
    work = sum(len(signal_panel) - lag for lag in policy.lags) * len(
        signal_panel.columns
    )
    if work > MAX_COMPARISON_CELLS:
        raise InvalidDelayEvidence(
            f"delay audit exceeds the {MAX_COMPARISON_CELLS} comparison-cell budget"
        )
    if any(is_bool_dtype(dtype) for dtype in signal_panel.dtypes):
        raise InvalidDelayEvidence("signal panel must not contain boolean columns")
    if any(is_bool_dtype(dtype) for dtype in forward_returns.dtypes):
        raise InvalidDelayEvidence(
            "forward-return panel must not contain boolean columns"
        )
    try:
        signal = signal_panel.astype(float)
        returns = forward_returns.reindex(columns=signal_panel.columns).astype(float)
    except (TypeError, ValueError) as exc:
        raise InvalidDelayEvidence(
            "signal and forward-return panels must be numeric"
        ) from exc
    if np.isinf(signal.to_numpy()).any() or np.isinf(returns.to_numpy()).any():
        raise InvalidDelayEvidence(
            "signal and forward-return panels must not contain infinity"
        )
    return signal, returns


def audit_signal_delay(
    signal_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    policy: SignalDelayPolicy = SignalDelayPolicy(),
) -> SignalDelayReport:
    """Measure whether cross-sectional IC survives an implementation delay."""

    _validate_policy(policy)
    signal, returns = _validate_panel(signal_panel, forward_returns, policy)
    lag_values: dict[int, list[float]] = {}
    for lag in policy.lags:
        values: list[float] = []
        for position in range(len(signal) - lag):
            pair = pd.concat(
                [signal.iloc[position], returns.iloc[position + lag]], axis=1
            ).dropna()
            if len(pair) < policy.minimum_assets_per_date:
                continue
            ic = information_coefficient(pair.iloc[:, 0], pair.iloc[:, 1])
            if math.isfinite(ic):
                values.append(ic)
        if len(values) < policy.minimum_observations:
            raise InvalidDelayEvidence(
                f"lag {lag} has {len(values)} observations; policy requires "
                f"{policy.minimum_observations}"
            )
        lag_values[lag] = values

    reference_mean = float(np.mean(lag_values[0]))
    direction = 1.0 if reference_mean >= 0 else -1.0
    points: list[SignalDelayPoint] = []
    for lag in policy.lags:
        values = np.asarray(lag_values[lag], dtype=float)
        mean_ic = float(np.mean(values))
        retention = (
            mean_ic * direction / abs(reference_mean)
            if abs(reference_mean) > 0.0
            else None
        )
        points.append(
            SignalDelayPoint(
                lag=lag,
                observations=len(values),
                mean_ic=mean_ic,
                median_ic=float(np.median(values)),
                positive_ic_rate=float(np.mean(values > 0.0)),
                oriented_ic_retention=retention,
            )
        )

    required_retention = next(
        point.oriented_ic_retention
        for point in points
        if point.lag == policy.required_lag
    )
    reasons: list[str] = []
    if abs(reference_mean) < policy.minimum_reference_abs_ic:
        reasons.append("reference_ic_below_minimum")
    if (
        required_retention is None
        or required_retention < policy.minimum_required_lag_retention
    ):
        reasons.append("required_lag_retention_below_minimum")
    return SignalDelayReport(
        schema_version=1,
        accepted=not reasons,
        reasons=tuple(reasons),
        required_lag=policy.required_lag,
        reference_mean_ic=reference_mean,
        reference_direction="positive" if direction > 0 else "negative",
        required_lag_retention=required_retention,
        policy=policy,
        points=tuple(points),
    )

"""Capacity and implementation-cost audit for factor portfolio targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


@dataclass(frozen=True, slots=True)
class ExecutionAuditPolicy:
    requested_capital_usd: float = 1_000_000.0
    max_participation_rate: float = 0.05
    commission_bps: float = 0.5
    impact_coefficient_bps: float = 10.0
    max_period_cost_bps: float = 20.0
    max_gross_leverage: float = 1.05
    dollar_neutral_tolerance: float = 1e-6
    min_periods: int = 20
    min_assets: int = 10
    min_active_assets: int = 2
    annualization_days: int = 252
    max_periods: int = 2_000
    max_assets: int = 5_000
    max_reported_periods: int = 100
    max_asset_id_chars: int = 128

    def __post_init__(self) -> None:
        _positive_finite("requested_capital_usd", self.requested_capital_usd)
        _rate("max_participation_rate", self.max_participation_rate, allow_zero=False)
        _non_negative_finite("commission_bps", self.commission_bps)
        _non_negative_finite("impact_coefficient_bps", self.impact_coefficient_bps)
        _positive_finite("max_period_cost_bps", self.max_period_cost_bps)
        _positive_finite("max_gross_leverage", self.max_gross_leverage)
        _non_negative_finite("dollar_neutral_tolerance", self.dollar_neutral_tolerance)
        for name in (
            "min_periods",
            "min_assets",
            "min_active_assets",
            "annualization_days",
            "max_periods",
            "max_assets",
            "max_reported_periods",
            "max_asset_id_chars",
        ):
            _positive_integer(name, getattr(self, name))
        if self.min_periods > self.max_periods:
            raise ValueError("min_periods cannot exceed max_periods")
        if self.min_assets > self.max_assets:
            raise ValueError("min_assets cannot exceed max_assets")
        if self.min_active_assets > self.max_assets:
            raise ValueError("min_active_assets cannot exceed max_assets")


@dataclass(frozen=True, slots=True)
class PeriodExecutionEvidence:
    timestamp: str
    turnover: float
    max_participation_rate: float
    capacity_usd: float | None
    trading_cost_bps: float
    borrow_cost_bps: float
    total_cost_bps: float
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["findings"] = list(self.findings)
        return value


@dataclass(frozen=True, slots=True)
class ExecutionAuditReport:
    passed: bool
    findings: tuple[str, ...]
    period_count: int
    asset_count: int
    requested_capital_usd: float
    minimum_capacity_usd: float
    median_capacity_usd: float
    maximum_participation_rate: float
    mean_turnover: float
    maximum_turnover: float
    mean_cost_bps: float
    maximum_cost_bps: float
    estimated_total_cost_usd: float
    input_sha256: str
    evidence_sha256: str
    periods_truncated: bool
    periods: tuple[PeriodExecutionEvidence, ...]
    policy: ExecutionAuditPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "findings": list(self.findings),
            "period_count": self.period_count,
            "asset_count": self.asset_count,
            "requested_capital_usd": self.requested_capital_usd,
            "minimum_capacity_usd": self.minimum_capacity_usd,
            "median_capacity_usd": self.median_capacity_usd,
            "maximum_participation_rate": self.maximum_participation_rate,
            "mean_turnover": self.mean_turnover,
            "maximum_turnover": self.maximum_turnover,
            "mean_cost_bps": self.mean_cost_bps,
            "maximum_cost_bps": self.maximum_cost_bps,
            "estimated_total_cost_usd": self.estimated_total_cost_usd,
            "input_sha256": self.input_sha256,
            "evidence_sha256": self.evidence_sha256,
            "periods_truncated": self.periods_truncated,
            "periods": [period.to_dict() for period in self.periods],
            "policy": asdict(self.policy),
        }


def audit_execution_capacity(
    target_weights: pd.DataFrame,
    adv_usd: pd.DataFrame,
    half_spread_bps: pd.DataFrame,
    *,
    annual_borrow_bps: pd.DataFrame | None = None,
    policy: ExecutionAuditPolicy | None = None,
) -> ExecutionAuditReport:
    """Audit whether target weights are executable under supplied liquidity evidence.

    Turnover is measured from cash for the first target and across the complete
    asset universe thereafter, so both new positions and liquidations are charged.
    Market impact uses a transparent square-root participation approximation.
    """
    active_policy = policy or ExecutionAuditPolicy()
    weights = _validated_frame("target_weights", target_weights, active_policy)
    adv = _validated_frame("adv_usd", adv_usd, active_policy)
    spreads = _validated_frame("half_spread_bps", half_spread_bps, active_policy)
    borrow = (
        _validated_frame("annual_borrow_bps", annual_borrow_bps, active_policy)
        if annual_borrow_bps is not None
        else pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    )
    _require_exact_alignment(weights, adv, spreads, borrow)
    if (adv <= 0.0).to_numpy().any():
        raise ValueError("adv_usd values must be positive")
    if (spreads < 0.0).to_numpy().any():
        raise ValueError("half_spread_bps values must be non-negative")
    if (borrow < 0.0).to_numpy().any():
        raise ValueError("annual_borrow_bps values must be non-negative")

    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1).abs()
    active = (weights.abs() > 0.0).sum(axis=1)
    if (gross <= 0.0).any():
        raise ValueError("every target period must contain a non-zero position")
    if (gross > active_policy.max_gross_leverage + 1e-12).any():
        raise ValueError("target gross leverage exceeds policy")
    if (net > active_policy.dollar_neutral_tolerance + 1e-12).any():
        raise ValueError("target weights violate dollar-neutral policy")
    if (active < active_policy.min_active_assets).any():
        raise ValueError("target period has too few active assets")

    previous = pd.Series(0.0, index=weights.columns)
    period_evidence: list[PeriodExecutionEvidence] = []
    capacities: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []

    for timestamp, current in weights.iterrows():
        trade_weights = (current - previous).abs()
        traded = trade_weights > 0.0
        turnover = float(trade_weights.sum())
        participation = trade_weights * active_policy.requested_capital_usd / adv.loc[timestamp]
        if not np.isfinite(participation.to_numpy()).all():
            raise ValueError("derived participation rates must be finite")
        max_participation = float(participation.max())
        capacity = (
            float((active_policy.max_participation_rate * adv.loc[timestamp, traded] / trade_weights.loc[traded]).min())
            if traded.any()
            else None
        )
        if capacity is not None and not math.isfinite(capacity):
            raise ValueError("derived capacity must be finite")
        execution_bps = (
            active_policy.commission_bps
            + spreads.loc[timestamp]
            + active_policy.impact_coefficient_bps * np.sqrt(participation)
        )
        trading_cost_bps = float((trade_weights * execution_bps).sum())
        short_weights = (-current).clip(lower=0.0)
        borrow_cost_bps = float((short_weights * borrow.loc[timestamp]).sum() / active_policy.annualization_days)
        total_cost_bps = trading_cost_bps + borrow_cost_bps
        if not all(math.isfinite(value) for value in (turnover, trading_cost_bps, borrow_cost_bps, total_cost_bps)):
            raise ValueError("derived execution costs must be finite")
        findings: list[str] = []
        if max_participation > active_policy.max_participation_rate + 1e-12:
            findings.append("PARTICIPATION_LIMIT_EXCEEDED")
        if capacity is not None and active_policy.requested_capital_usd > capacity + 1e-9:
            findings.append("REQUESTED_CAPITAL_ABOVE_CAPACITY")
        if total_cost_bps > active_policy.max_period_cost_bps + 1e-12:
            findings.append("PERIOD_COST_LIMIT_EXCEEDED")
        period_evidence.append(
            PeriodExecutionEvidence(
                timestamp=timestamp.isoformat(),
                turnover=turnover,
                max_participation_rate=max_participation,
                capacity_usd=capacity,
                trading_cost_bps=trading_cost_bps,
                borrow_cost_bps=borrow_cost_bps,
                total_cost_bps=total_cost_bps,
                findings=tuple(findings),
            )
        )
        if capacity is not None:
            capacities.append(capacity)
        turnovers.append(turnover)
        costs.append(total_cost_bps)
        previous = current

    findings = tuple(sorted({finding for period in period_evidence for finding in period.findings}))
    reported_periods = _bounded_period_evidence(period_evidence, active_policy.max_reported_periods)
    input_sha256 = _input_digest(weights, adv, spreads, borrow)
    report_payload = {
        "passed": not findings,
        "findings": list(findings),
        "period_count": len(weights),
        "asset_count": len(weights.columns),
        "requested_capital_usd": active_policy.requested_capital_usd,
        "minimum_capacity_usd": min(capacities),
        "median_capacity_usd": float(np.median(capacities)),
        "maximum_participation_rate": max(period.max_participation_rate for period in period_evidence),
        "mean_turnover": float(np.mean(turnovers)),
        "maximum_turnover": max(turnovers),
        "mean_cost_bps": float(np.mean(costs)),
        "maximum_cost_bps": max(costs),
        "estimated_total_cost_usd": float(sum(costs) * active_policy.requested_capital_usd / 10_000.0),
        "input_sha256": input_sha256,
        "periods_truncated": len(period_evidence) > len(reported_periods),
        "periods": [period.to_dict() for period in reported_periods],
        "policy": asdict(active_policy),
    }
    evidence_sha256 = hashlib.sha256(_canonical_json(report_payload)).hexdigest()
    return ExecutionAuditReport(
        passed=not findings,
        findings=findings,
        period_count=len(weights),
        asset_count=len(weights.columns),
        requested_capital_usd=active_policy.requested_capital_usd,
        minimum_capacity_usd=min(capacities),
        median_capacity_usd=float(np.median(capacities)),
        maximum_participation_rate=max(period.max_participation_rate for period in period_evidence),
        mean_turnover=float(np.mean(turnovers)),
        maximum_turnover=max(turnovers),
        mean_cost_bps=float(np.mean(costs)),
        maximum_cost_bps=max(costs),
        estimated_total_cost_usd=float(sum(costs) * active_policy.requested_capital_usd / 10_000.0),
        input_sha256=input_sha256,
        evidence_sha256=evidence_sha256,
        periods_truncated=len(period_evidence) > len(reported_periods),
        periods=reported_periods,
        policy=active_policy,
    )


def _validated_frame(
    name: str,
    frame: pd.DataFrame | None,
    policy: ExecutionAuditPolicy,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    if len(frame) < policy.min_periods or len(frame) > policy.max_periods:
        raise ValueError(f"{name} period count is outside policy bounds")
    if len(frame.columns) < policy.min_assets or len(frame.columns) > policy.max_assets:
        raise ValueError(f"{name} asset count is outside policy bounds")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} index must be a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError(f"{name} timestamps must be timezone-aware")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be unique and strictly increasing")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} asset identifiers must be unique")
    for column in frame.columns:
        if not isinstance(column, str) or not column or column != column.strip():
            raise ValueError(f"{name} asset identifiers must be normalized non-empty strings")
        if len(column) > policy.max_asset_id_chars:
            raise ValueError(f"{name} asset identifier exceeds policy length")
        if is_bool_dtype(frame[column].dtype) or not is_numeric_dtype(frame[column].dtype):
            raise TypeError(f"{name} values must be numeric and non-boolean")
    values = frame.astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{name} values must be finite")
    return values


def _require_exact_alignment(*frames: pd.DataFrame) -> None:
    first = frames[0]
    if any(not frame.index.equals(first.index) for frame in frames[1:]):
        raise ValueError("all evidence frames must have identical ordered timestamps")
    if any(not frame.columns.equals(first.columns) for frame in frames[1:]):
        raise ValueError("all evidence frames must have identical ordered assets")


def _bounded_period_evidence(
    periods: list[PeriodExecutionEvidence],
    limit: int,
) -> tuple[PeriodExecutionEvidence, ...]:
    violations = [period for period in periods if period.findings]
    passing = [period for period in periods if not period.findings]
    return tuple((violations + passing)[:limit])


def _input_digest(*frames: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for frame_number, frame in enumerate(frames):
        digest.update(f"frame:{frame_number}\0".encode())
        digest.update(_canonical_json(list(frame.columns)))
        digest.update(b"\0")
        for timestamp, row in frame.iterrows():
            digest.update(timestamp.isoformat().encode())
            digest.update(b"\0")
            for asset, value in row.items():
                digest.update(asset.encode())
                digest.update(b"\0")
                digest.update(format(float(value), ".17g").encode())
                digest.update(b"\0")
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _non_negative_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _positive_finite(name: str, value: float) -> None:
    _non_negative_finite(name, value)
    if value == 0.0:
        raise ValueError(f"{name} must be positive")


def _rate(name: str, value: float, *, allow_zero: bool) -> None:
    _non_negative_finite(name, value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not lower_ok or value > 1.0:
        raise ValueError(f"{name} must be within its unit interval")


def _frame_from_payload(payload: dict[str, Any], key: str) -> pd.DataFrame:
    timestamps = [pd.Timestamp(value) for value in payload["timestamps"]]
    if any(timestamp.tzinfo is None for timestamp in timestamps):
        raise ValueError("timestamps must include an explicit timezone offset")
    timestamps = pd.DatetimeIndex(timestamps).tz_convert("UTC")
    return pd.DataFrame(payload[key], index=timestamps, columns=payload["assets"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit factor targets for capacity and implementation cost")
    parser.add_argument("input", type=Path, help="JSON execution-evidence artifact")
    parser.add_argument("--require-pass", action="store_true", help="return 2 when policy rejects the artifact")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        report = audit_execution_capacity(
            _frame_from_payload(payload, "target_weights"),
            _frame_from_payload(payload, "adv_usd"),
            _frame_from_payload(payload, "half_spread_bps"),
            annual_borrow_bps=(
                _frame_from_payload(payload, "annual_borrow_bps") if "annual_borrow_bps" in payload else None
            ),
            policy=ExecutionAuditPolicy(**payload.get("policy", {})),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 2 if args.require_pass and not report.passed else 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


def winsorized_zscore(series: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    lo, hi = series.quantile([lower, upper])
    clipped = series.clip(lo, hi)
    std = clipped.std(ddof=0)
    return (clipped - clipped.mean()) / std if std > 0 else pd.Series(0.0, index=series.index)


@dataclass(frozen=True)
class NeutralizationDiagnostics:
    observations: int
    exposure_count: int
    matrix_rank: int
    condition_number: float
    absolute_residual_mean: float
    maximum_absolute_exposure_correlation: float

    def as_dict(self) -> dict[str, int | float]:
        return self.__dict__.copy()


def neutralize_with_diagnostics(
    values: pd.Series,
    exposures: pd.DataFrame,
    *,
    max_condition_number: float = 1e8,
    orthogonality_tolerance: float = 1e-10,
) -> tuple[pd.Series, NeutralizationDiagnostics]:
    """Residualize a cross-section and prove the result is numerically neutral."""
    if not isinstance(values, pd.Series) or not isinstance(exposures, pd.DataFrame):
        raise TypeError("values and exposures must be a Series and DataFrame")
    if exposures.empty or exposures.shape[1] > 32:
        raise ValueError("exposures must contain between 1 and 32 columns")
    if values.index.has_duplicates or exposures.index.has_duplicates:
        raise ValueError("asset identifiers must be unique")
    if not values.index.equals(exposures.index):
        raise ValueError("values and exposures must have identical ordered assets")
    if exposures.columns.has_duplicates:
        raise ValueError("exposure names must be unique")
    if len(values) <= exposures.shape[1] + 1:
        raise ValueError("neutralization requires more observations than regressors")
    for name, value in (
        ("max_condition_number", max_condition_number),
        ("orthogonality_tolerance", orthogonality_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    if is_bool_dtype(values.dtype) or any(
        is_bool_dtype(exposures[column].dtype) for column in exposures.columns
    ):
        raise TypeError("values and exposures must be numeric and non-boolean")
    try:
        target = values.astype(float)
        design = exposures.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("values and exposures must be numeric") from exc
    if (
        not np.isfinite(target.to_numpy()).all()
        or not np.isfinite(design.to_numpy()).all()
    ):
        raise ValueError("values and exposures must be finite")

    matrix = np.column_stack((np.ones(len(design)), design.to_numpy()))
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != matrix.shape[1]:
        raise ValueError("neutralization design matrix is rank deficient")
    condition_number = float(np.linalg.cond(matrix))
    if not isfinite(condition_number) or condition_number > max_condition_number:
        raise ValueError("neutralization design matrix is ill-conditioned")

    beta, *_ = np.linalg.lstsq(matrix, target.to_numpy(), rcond=None)
    residual_values = target.to_numpy() - matrix @ beta
    residual = pd.Series(residual_values, index=values.index, name=values.name)
    correlations = [
        abs(float(np.corrcoef(residual_values, design[column].to_numpy())[0, 1]))
        for column in design.columns
    ]
    max_correlation = max(correlations, default=0.0)
    absolute_mean = abs(float(residual.mean()))
    if not isfinite(max_correlation) or max_correlation > orthogonality_tolerance:
        raise RuntimeError(
            "neutralization residual failed the exposure-orthogonality check"
        )
    if not isfinite(absolute_mean) or absolute_mean > orthogonality_tolerance:
        raise RuntimeError(
            "neutralization residual failed the intercept-orthogonality check"
        )

    return residual, NeutralizationDiagnostics(
        observations=len(values),
        exposure_count=exposures.shape[1],
        matrix_rank=rank,
        condition_number=condition_number,
        absolute_residual_mean=absolute_mean,
        maximum_absolute_exposure_correlation=max_correlation,
    )


def neutralize(values: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    residual, _ = neutralize_with_diagnostics(values, exposures)
    return residual


def information_coefficient(signal: pd.Series, forward_return: pd.Series) -> float:
    aligned = pd.concat([signal, forward_return], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman"))


@dataclass(frozen=True)
class BacktestResult:
    gross_return: float
    net_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    average_turnover: float
    observations: int

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def long_short_weights(signal: pd.Series, quantile: float = 0.20) -> pd.Series:
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5")
    n = len(signal)
    count = max(1, int(n * quantile))
    ranking = signal.rank(method="first")
    long_names = ranking.nlargest(count).index
    short_names = ranking.nsmallest(count).index
    weights = pd.Series(0.0, index=signal.index)
    weights.loc[long_names] = 0.5 / count
    weights.loc[short_names] = -0.5 / count
    return weights


def backtest_cross_sectional_factor(
    signal_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    quantile: float = 0.20,
    cost_bps: float = 5.0,
) -> tuple[BacktestResult, pd.DataFrame]:
    dates = signal_panel.index.intersection(forward_returns.index)
    prev_weights: pd.Series | None = None
    rows: list[dict[str, float | pd.Timestamp]] = []
    cost_rate = cost_bps / 10_000.0

    for date in dates:
        signal = signal_panel.loc[date].dropna()
        returns = forward_returns.loc[date].reindex(signal.index).dropna()
        signal = signal.reindex(returns.index)
        if len(signal) < 5:
            continue
        weights = long_short_weights(signal, quantile)
        turnover = float((weights - (prev_weights.reindex(weights.index).fillna(0.0) if prev_weights is not None else 0.0)).abs().sum())
        gross = float((weights * returns).sum())
        net = gross - turnover * cost_rate
        rows.append({"date": date, "gross": gross, "net": net, "turnover": turnover})
        prev_weights = weights

    history = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["gross", "net", "turnover"])
    if history.empty:
        result = BacktestResult(0, 0, 0, 0, 0, 0, 0, 0)
        return result, history
    wealth = (1.0 + history["net"]).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    ann_return = float(history["net"].mean() * 252)
    ann_vol = float(history["net"].std(ddof=1) * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    result = BacktestResult(
        gross_return=float((1.0 + history["gross"]).prod() - 1.0),
        net_return=float(wealth.iloc[-1] - 1.0),
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe=sharpe,
        max_drawdown=float(drawdown.min()),
        average_turnover=float(history["turnover"].mean()),
        observations=len(history),
    )
    return result, history

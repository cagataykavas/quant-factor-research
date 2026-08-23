from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def winsorized_zscore(series: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    lo, hi = series.quantile([lower, upper])
    clipped = series.clip(lo, hi)
    std = clipped.std(ddof=0)
    return (clipped - clipped.mean()) / std if std > 0 else pd.Series(0.0, index=series.index)


def neutralize(values: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    x = exposures.copy().astype(float)
    x.insert(0, "intercept", 1.0)
    beta, *_ = np.linalg.lstsq(x.to_numpy(), values.to_numpy(), rcond=None)
    residual = values.to_numpy() - x.to_numpy() @ beta
    return pd.Series(residual, index=values.index, name=values.name)


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

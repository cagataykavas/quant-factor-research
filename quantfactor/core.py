from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


def winsorized_zscore(
    series: pd.Series, lower: float = 0.02, upper: float = 0.98
) -> pd.Series:
    lo, hi = series.quantile([lower, upper])
    clipped = series.clip(lo, hi)
    std = clipped.std(ddof=0)
    return (
        (clipped - clipped.mean()) / std
        if std > 0
        else pd.Series(0.0, index=series.index)
    )


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
class LifecycleCost:
    turnover: float
    return_deduction: float


@dataclass(frozen=True)
class TransactionCostBreakdown:
    cost_bps: float
    terminal_liquidation: bool
    entry: LifecycleCost
    rebalance: LifecycleCost
    exit: LifecycleCost
    terminal: LifecycleCost
    total: LifecycleCost


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
    transaction_costs: TransactionCostBreakdown

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    liquidate_at_end: bool = False,
) -> tuple[BacktestResult, pd.DataFrame]:
    if isinstance(cost_bps, bool) or not isinstance(cost_bps, Real):
        raise TypeError("cost_bps must be a finite non-negative number")
    cost_bps = float(cost_bps)
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be a finite non-negative number")
    if not isinstance(liquidate_at_end, bool):
        raise TypeError("liquidate_at_end must be a boolean")

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
        if prev_weights is None:
            entry_turnover = float(weights.abs().sum())
            rebalance_turnover = 0.0
            exit_turnover = 0.0
        else:
            holdings = weights.index.union(prev_weights.index, sort=False)
            current = weights.reindex(holdings, fill_value=0.0)
            previous = prev_weights.reindex(holdings, fill_value=0.0)
            total_rebalance = float((current - previous).abs().sum())
            exited = prev_weights.index.difference(weights.index)
            exit_turnover = float(prev_weights.reindex(exited).abs().sum())
            entry_turnover = 0.0
            rebalance_turnover = max(0.0, total_rebalance - exit_turnover)
        terminal_turnover = 0.0
        turnover = entry_turnover + rebalance_turnover + exit_turnover
        gross = float((weights * returns).sum())
        transaction_cost = turnover * cost_rate
        net = gross - transaction_cost
        rows.append(
            {
                "date": date,
                "gross": gross,
                "net": net,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "entry_turnover": entry_turnover,
                "rebalance_turnover": rebalance_turnover,
                "exit_turnover": exit_turnover,
                "terminal_turnover": terminal_turnover,
            }
        )
        prev_weights = weights

    if rows and liquidate_at_end:
        terminal_turnover = float(prev_weights.abs().sum())
        terminal_cost = terminal_turnover * cost_rate
        rows[-1]["terminal_turnover"] = terminal_turnover
        rows[-1]["turnover"] += terminal_turnover
        rows[-1]["transaction_cost"] += terminal_cost
        rows[-1]["net"] -= terminal_cost

    history_columns = [
        "gross",
        "net",
        "turnover",
        "transaction_cost",
        "entry_turnover",
        "rebalance_turnover",
        "exit_turnover",
        "terminal_turnover",
    ]
    history = (
        pd.DataFrame(rows).set_index("date")
        if rows
        else pd.DataFrame(columns=history_columns)
    )
    transaction_costs = _transaction_cost_breakdown(
        history, cost_bps=cost_bps, terminal_liquidation=liquidate_at_end
    )
    if history.empty:
        result = BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, transaction_costs)
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
        transaction_costs=transaction_costs,
    )
    return result, history


def _transaction_cost_breakdown(
    history: pd.DataFrame, *, cost_bps: float, terminal_liquidation: bool
) -> TransactionCostBreakdown:
    cost_rate = cost_bps / 10_000.0

    def stage(name: str) -> LifecycleCost:
        turnover = float(history[f"{name}_turnover"].sum())
        return LifecycleCost(turnover, turnover * cost_rate)

    entry = stage("entry")
    rebalance = stage("rebalance")
    exit_cost = stage("exit")
    terminal = stage("terminal")
    total_turnover = sum(
        item.turnover for item in (entry, rebalance, exit_cost, terminal)
    )
    return TransactionCostBreakdown(
        cost_bps=cost_bps,
        terminal_liquidation=terminal_liquidation,
        entry=entry,
        rebalance=rebalance,
        exit=exit_cost,
        terminal=terminal,
        total=LifecycleCost(total_turnover, total_turnover * cost_rate),
    )

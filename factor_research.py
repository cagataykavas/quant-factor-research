from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
    lo, hi = s.quantile([q, 1 - q])
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / (std if std > 0 else 1.0)


def build_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Expected columns: date, asset, close, volume."""
    x = df.sort_values(["asset", "date"]).copy()
    g = x.groupby("asset", group_keys=False)
    x["ret_1d"] = g["close"].pct_change()
    x["momentum_20"] = g["close"].pct_change(20)
    x["reversal_5"] = -g["close"].pct_change(5)
    x["volatility_20"] = g["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    x["volume_ratio"] = x["volume"] / g["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    x["future_ret_5"] = g["close"].shift(-5) / x["close"] - 1
    return x


def cross_sectional_scores(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    out = df.copy()
    out["factor_score"] = out.groupby("date")[factor].transform(lambda s: zscore(winsorize(s)))
    return out


def information_coefficient(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date").apply(
        lambda x: x["factor_score"].corr(x["future_ret_5"], method="spearman"),
        include_groups=False,
    ).dropna()


def quantile_backtest(df: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    x = df.dropna(subset=["factor_score", "future_ret_5"]).copy()
    x["bucket"] = x.groupby("date")["factor_score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), quantiles, labels=False, duplicates="drop")
    )
    return x.groupby(["date", "bucket"], as_index=False)["future_ret_5"].mean()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="B")
    rows = []
    for asset_id in range(30):
        returns = rng.normal(0.0003, 0.015, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = rng.lognormal(13, 0.35, len(dates))
        rows.extend(zip(dates, [f"ASSET_{asset_id:02d}"] * len(dates), prices, volumes))
    data = pd.DataFrame(rows, columns=["date", "asset", "close", "volume"])
    factors = cross_sectional_scores(build_factors(data), "momentum_20")
    ic = information_coefficient(factors)
    print({"mean_ic": round(float(ic.mean()), 4), "ic_ir": round(float(ic.mean() / ic.std()), 4)})
    print(quantile_backtest(factors).tail())

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import (
    backtest_cross_sectional_factor,
    information_coefficient,
    neutralize_with_diagnostics,
    winsorized_zscore,
)


class FactorResearchEngine:
    def _prepare_signal(
        self, raw_signal: pd.DataFrame, size_exposure: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, list[dict[str, int | float]]]:
        if size_exposure is not None and (
            not raw_signal.index.equals(size_exposure.index)
            or not raw_signal.columns.equals(size_exposure.columns)
        ):
            raise ValueError(
                "signal and exposure panels must have identical ordered axes"
            )
        rows: list[pd.Series] = []
        diagnostics: list[dict[str, int | float]] = []
        for date in raw_signal.index:
            values = winsorized_zscore(raw_signal.loc[date])
            if size_exposure is not None:
                exposure = pd.DataFrame(
                    {"size": size_exposure.loc[date].reindex(values.index)}
                )
                values, _ = neutralize_with_diagnostics(values, exposure)
                values = winsorized_zscore(values)
                # Winsorizing residuals is nonlinear and can reintroduce exposure.
                # Residualize again so the diagnostic describes the final signal.
                values, audit = neutralize_with_diagnostics(values, exposure)
                diagnostics.append(audit.as_dict())
            values.name = date
            rows.append(values)
        return pd.DataFrame(rows), diagnostics

    def prepare_signal(
        self, raw_signal: pd.DataFrame, size_exposure: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        signal, _ = self._prepare_signal(raw_signal, size_exposure)
        return signal

    def evaluate(
        self,
        raw_signal: pd.DataFrame,
        forward_returns: pd.DataFrame,
        *,
        size_exposure: pd.DataFrame | None = None,
        quantile: float = 0.20,
        cost_bps: float = 5.0,
    ) -> dict[str, Any]:
        signal, neutralization_rows = self._prepare_signal(raw_signal, size_exposure)
        ics: list[float] = []
        for date in signal.index.intersection(forward_returns.index):
            ic = information_coefficient(signal.loc[date], forward_returns.loc[date])
            if np.isfinite(ic):
                ics.append(ic)
        result, history = backtest_cross_sectional_factor(
            signal, forward_returns, quantile=quantile, cost_bps=cost_bps
        )
        ic_mean = float(np.mean(ics)) if ics else 0.0
        ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        ic_ir = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else 0.0
        cost_sensitivity = {}
        for cost in (0.0, 2.0, 5.0, 10.0, 20.0):
            cost_result, _ = backtest_cross_sectional_factor(
                signal, forward_returns, quantile=quantile, cost_bps=cost
            )
            cost_sensitivity[str(cost)] = cost_result.net_return
        return {
            "mean_daily_ic": ic_mean,
            "ic_information_ratio": float(ic_ir),
            "positive_ic_rate": float(np.mean(np.asarray(ics) > 0)) if ics else 0.0,
            "backtest": result.as_dict(),
            "transaction_cost_sensitivity": cost_sensitivity,
            "neutralization_diagnostics": {
                "dates_audited": len(neutralization_rows),
                "maximum_condition_number": max(
                    (row["condition_number"] for row in neutralization_rows),
                    default=None,
                ),
                "maximum_absolute_residual_mean": max(
                    (row["absolute_residual_mean"] for row in neutralization_rows),
                    default=None,
                ),
                "maximum_absolute_exposure_correlation": max(
                    (
                        row["maximum_absolute_exposure_correlation"]
                        for row in neutralization_rows
                    ),
                    default=None,
                ),
            },
            "history": [
                {
                    "date": str(index),
                    **{column: float(value) for column, value in row.items()},
                }
                for index, row in history.iterrows()
            ],
        }

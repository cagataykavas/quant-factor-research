from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from quantfactor.core import backtest_cross_sectional_factor
from quantfactor.engine import FactorResearchEngine
from quantfactor.synthetic import synthetic_factor_panel


def test_factor_pipeline_produces_ic_and_backtest_history():
    signal, forward, size = synthetic_factor_panel(dates=220, assets=50, seed=42)
    result = FactorResearchEngine().evaluate(signal, forward, size_exposure=size, cost_bps=5.0)
    assert result["backtest"]["observations"] > 100
    assert -1.0 <= result["mean_daily_ic"] <= 1.0
    assert 0.0 <= result["positive_ic_rate"] <= 1.0
    assert len(result["transaction_cost_sensitivity"]) == 5


def test_higher_cost_does_not_improve_same_backtest():
    signal, forward, size = synthetic_factor_panel(dates=160, assets=40, seed=7)
    result = FactorResearchEngine().evaluate(signal, forward, size_exposure=size)
    sensitivity = result["transaction_cost_sensitivity"]
    assert sensitivity["20.0"] <= sensitivity["0.0"] + 1e-12


def test_turnover_charges_exits_when_assets_leave_the_eligible_universe():
    dates = pd.date_range("2025-01-01", periods=2)
    signal = pd.DataFrame(
        [
            {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0},
            {"B": 1.0, "C": 2.0, "D": 3.0, "E": 4.0, "F": 5.0, "G": 6.0},
        ],
        index=dates,
    )
    forward_returns = pd.DataFrame(0.0, index=dates, columns=signal.columns)

    result, history = backtest_cross_sectional_factor(
        signal, forward_returns, quantile=0.2, cost_bps=100.0
    )

    assert history["turnover"].tolist() == pytest.approx([1.0, 2.0])
    assert history["net"].tolist() == pytest.approx([-0.01, -0.02])
    assert result.net_return == pytest.approx((1.0 - 0.01) * (1.0 - 0.02) - 1.0)


def test_demo_api():
    response = TestClient(app).get("/demo?dates=150&assets=35&seed=2")
    assert response.status_code == 200
    assert response.json()["backtest"]["observations"] > 50

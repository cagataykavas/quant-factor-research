from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfactor.delay import (
    InvalidDelayEvidence,
    SignalDelayPolicy,
    audit_signal_delay,
)
from quantfactor.engine import FactorResearchEngine
from quantfactor.synthetic import synthetic_factor_panel


def panels(rows: int = 90, assets: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    index = pd.date_range("2025-01-02", periods=rows, freq="B")
    columns = [f"asset-{number}" for number in range(assets)]
    signal = pd.DataFrame(rng.normal(size=(rows, assets)), index=index, columns=columns)
    return signal, signal.copy()


def policy(**overrides) -> SignalDelayPolicy:
    values = {
        "lags": (0, 1, 2),
        "required_lag": 1,
        "minimum_assets_per_date": 5,
        "minimum_observations": 60,
        "minimum_reference_abs_ic": 0.01,
        "minimum_required_lag_retention": 0.5,
    }
    values.update(overrides)
    return SignalDelayPolicy(**values)


def test_rejects_signal_that_disappears_after_execution_delay() -> None:
    signal, returns = panels()

    report = audit_signal_delay(signal, returns, policy())

    assert not report.accepted
    assert report.reference_mean_ic == pytest.approx(1.0)
    assert report.required_lag_retention < 0.5
    assert report.reasons == ("required_lag_retention_below_minimum",)


def test_accepts_persistent_signal_and_reports_each_lag() -> None:
    signal, _ = panels()
    persistent = pd.DataFrame(
        np.tile(np.arange(signal.shape[1]), (len(signal), 1)),
        index=signal.index,
        columns=signal.columns,
        dtype=float,
    )

    report = audit_signal_delay(persistent, persistent.copy(), policy())

    assert report.accepted
    assert [point.lag for point in report.points] == [0, 1, 2]
    assert report.required_lag_retention == pytest.approx(1.0)
    assert all(point.observations >= 88 for point in report.points)


def test_sign_flip_does_not_pass_absolute_retention() -> None:
    signal, _ = panels()
    base = np.arange(signal.shape[1], dtype=float)
    alternating = pd.DataFrame(
        [base if index % 2 == 0 else -base for index in range(len(signal))],
        index=signal.index,
        columns=signal.columns,
    )

    report = audit_signal_delay(alternating, alternating.copy(), policy(lags=(0, 1)))

    assert not report.accepted
    assert report.required_lag_retention == pytest.approx(-1.0)


@pytest.mark.parametrize("mutation", ["dates", "assets", "infinity"])
def test_fails_closed_on_misaligned_or_non_finite_evidence(mutation: str) -> None:
    signal, returns = panels()
    if mutation == "dates":
        returns = returns.iloc[1:]
    elif mutation == "assets":
        returns = returns.drop(columns=returns.columns[-1])
    else:
        signal.iloc[0, 0] = float("inf")

    with pytest.raises(InvalidDelayEvidence):
        audit_signal_delay(signal, returns, policy())


def test_engine_exposes_bounded_delay_evidence() -> None:
    signal, returns, size = synthetic_factor_panel(dates=100, assets=20, seed=4)

    result = FactorResearchEngine().evaluate(signal, returns, size_exposure=size)
    delay = result["signal_delay_audit"]

    assert delay["schema_version"] == 1
    assert delay["required_lag"] == 1
    assert len(delay["points"]) == 4
    assert all(point["observations"] >= 60 for point in delay["points"])

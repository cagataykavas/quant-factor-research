from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantfactor.core import neutralize_with_diagnostics
from quantfactor.engine import FactorResearchEngine
from quantfactor.synthetic import synthetic_factor_panel


def test_neutralization_reports_verified_orthogonality() -> None:
    assets = pd.Index([f"asset-{index}" for index in range(20)])
    exposure = pd.DataFrame({"size": np.linspace(-2.0, 2.0, len(assets))}, index=assets)
    values = pd.Series(
        3.0 * exposure["size"] + np.sin(np.arange(len(assets))), index=assets
    )

    residual, audit = neutralize_with_diagnostics(values, exposure)

    assert audit.observations == 20
    assert audit.exposure_count == 1
    assert audit.matrix_rank == 2
    assert audit.absolute_residual_mean < 1e-12
    assert audit.maximum_absolute_exposure_correlation < 1e-12
    assert abs(float(residual.corr(exposure["size"]))) < 1e-12


def test_engine_publishes_bounded_neutralization_summary() -> None:
    signal, forward, size = synthetic_factor_panel(dates=40, assets=30, seed=7)

    result = FactorResearchEngine().evaluate(signal, forward, size_exposure=size)
    audit = result["neutralization_diagnostics"]

    assert audit["dates_audited"] == 40
    assert audit["maximum_condition_number"] >= 1.0
    assert audit["maximum_absolute_residual_mean"] < 1e-10
    assert audit["maximum_absolute_exposure_correlation"] < 1e-10
    prepared = FactorResearchEngine().prepare_signal(signal, size)
    assert (
        max(
            abs(float(prepared.loc[date].corr(size.loc[date])))
            for date in prepared.index
        )
        < 1e-10
    )
    json.dumps(result, allow_nan=False)


def test_engine_requires_exact_panel_alignment() -> None:
    signal, forward, size = synthetic_factor_panel(dates=20, assets=15, seed=3)

    with pytest.raises(ValueError, match="identical ordered axes"):
        FactorResearchEngine().evaluate(
            signal,
            forward,
            size_exposure=size[size.columns[::-1]],
        )


def test_rank_deficient_exposures_fail_closed() -> None:
    index = pd.Index([f"asset-{value}" for value in range(12)])
    values = pd.Series(np.linspace(-1.0, 1.0, len(index)), index=index)
    exposure = np.linspace(0.0, 1.0, len(index))
    exposures = pd.DataFrame({"first": exposure, "duplicate": exposure}, index=index)

    with pytest.raises(ValueError, match="rank deficient"):
        neutralize_with_diagnostics(values, exposures)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.iloc[::-1], "identical ordered assets"),
        (lambda frame: frame.assign(size=float("nan")), "finite"),
        (lambda frame: frame.assign(size=True), "non-boolean"),
        (
            lambda frame: frame.rename(columns={"size": "duplicate"}).assign(
                duplicate_2=1.0
            ),
            "rank deficient",
        ),
    ],
)
def test_malformed_exposure_evidence_is_rejected(mutate, message: str) -> None:
    index = pd.Index([f"asset-{value}" for value in range(12)])
    values = pd.Series(np.linspace(-1.0, 1.0, len(index)), index=index)
    exposures = pd.DataFrame({"size": np.linspace(0.0, 1.0, len(index))}, index=index)

    with pytest.raises((TypeError, ValueError), match=message):
        neutralize_with_diagnostics(values, mutate(exposures))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_condition_number": 0.0},
        {"max_condition_number": float("nan")},
        {"orthogonality_tolerance": True},
    ],
)
def test_invalid_numeric_policy_is_rejected(kwargs: dict[str, object]) -> None:
    index = pd.Index([f"asset-{value}" for value in range(12)])
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    exposures = pd.DataFrame({"size": np.linspace(-1.0, 1.0, len(index))}, index=index)

    with pytest.raises((TypeError, ValueError)):
        neutralize_with_diagnostics(values, exposures, **kwargs)

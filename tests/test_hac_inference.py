from __future__ import annotations

import numpy as np
import pytest

from quantfactor.inference import automatic_newey_west_lag, hac_mean_inference


def test_lag_zero_matches_population_variance_standard_error() -> None:
    values = [-0.2, -0.1, 0.0, 0.1, 0.2]
    report = hac_mean_inference(values, lag=0)

    expected = float(np.std(values, ddof=0) / np.sqrt(len(values)))
    assert report.hac_standard_error == pytest.approx(expected)
    assert report.lag == 0
    assert report.bandwidth_method == "explicit"
    assert report.mean == pytest.approx(0)
    assert report.two_sided_p_value == pytest.approx(1)


def test_hac_standard_error_reflects_positive_serial_dependence() -> None:
    rng = np.random.default_rng(12)
    values = np.empty(1200)
    values[0] = 0.01
    for index in range(1, len(values)):
        values[index] = 0.85 * values[index - 1] + rng.normal(0.001, 0.02)

    report = hac_mean_inference(values, lag=12)

    assert report.hac_standard_error > report.naive_standard_error * 2
    assert report.standard_error_ratio > 2
    assert 0 <= report.two_sided_p_value <= 1


def test_automatic_bandwidth_is_recorded_and_reproducible() -> None:
    values = np.linspace(-0.2, 0.3, 250)
    first = hac_mean_inference(values)
    second = hac_mean_inference(values)

    assert first == second
    assert first.lag == automatic_newey_west_lag(250)
    assert first.bandwidth_method == "automatic"
    assert first.observations == 250


def test_zero_variance_series_does_not_emit_infinite_statistics() -> None:
    report = hac_mean_inference([0.0] * 20)

    assert report.hac_standard_error == 0
    assert report.standard_error_ratio is None
    assert report.t_statistic is None
    assert report.two_sided_p_value is None


@pytest.mark.parametrize(
    "values, lag, exception, message",
    [
        ([0.1, 0.2], None, ValueError, "at least three"),
        ([0.1, float("nan"), 0.2], None, ValueError, "finite"),
        ([0.1, 1.1, 0.2], None, ValueError, r"\[-1, 1\]"),
        ([0.1, 0.2, 0.3], -1, ValueError, "between zero"),
        ([0.1, 0.2, 0.3], 3, ValueError, "between zero"),
        ([0.1, 0.2, 0.3], True, TypeError, "integer"),
    ],
)
def test_invalid_inference_contract_fails_closed(
    values: list[float], lag: int | None, exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        hac_mean_inference(values, lag=lag)

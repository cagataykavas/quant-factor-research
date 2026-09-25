from __future__ import annotations

import json
from dataclasses import asdict, replace

import pandas as pd
import pytest

from quantfactor.execution_audit import (
    ExecutionAuditPolicy,
    audit_execution_capacity,
    main,
)


def evidence(*, periods: int = 4, assets: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2026-01-05", periods=periods, freq="B", tz="UTC")
    columns = [f"asset-{number}" for number in range(assets)]
    rows = []
    for period in range(periods):
        shift = period % assets
        row = [0.0] * assets
        row[shift] = 0.5
        row[(shift + 1) % assets] = -0.5
        rows.append(row)
    weights = pd.DataFrame(rows, index=index, columns=columns)
    adv = pd.DataFrame(100_000_000.0, index=index, columns=columns)
    spreads = pd.DataFrame(1.0, index=index, columns=columns)
    borrow = pd.DataFrame(252.0, index=index, columns=columns)
    return weights, adv, spreads, borrow


def policy(**changes: object) -> ExecutionAuditPolicy:
    return replace(
        ExecutionAuditPolicy(min_periods=4, min_assets=4, max_period_cost_bps=100.0),
        **changes,
    )


def test_liquid_targets_pass_with_cost_and_capacity_evidence() -> None:
    weights, adv, spreads, borrow = evidence()
    report = audit_execution_capacity(weights, adv, spreads, annual_borrow_bps=borrow, policy=policy())

    assert report.passed
    assert report.findings == ()
    assert report.period_count == 4
    assert report.asset_count == 4
    assert report.minimum_capacity_usd == pytest.approx(5_000_000.0)
    assert report.maximum_participation_rate == pytest.approx(0.01)
    assert report.periods[0].borrow_cost_bps == pytest.approx(0.5)


def test_initial_entry_and_removed_assets_are_charged_to_turnover() -> None:
    weights, adv, spreads, _ = evidence()
    weights.iloc[1] = [0.0, 0.0, 0.5, -0.5]

    report = audit_execution_capacity(weights, adv, spreads, policy=policy())

    assert report.periods[0].turnover == pytest.approx(1.0)
    assert report.periods[1].turnover == pytest.approx(2.0)


def test_rejects_requested_capital_above_participation_capacity() -> None:
    weights, adv, spreads, _ = evidence()
    adv.iloc[2, 2] = 1_000_000.0

    report = audit_execution_capacity(weights, adv, spreads, policy=policy())

    assert not report.passed
    assert "PARTICIPATION_LIMIT_EXCEEDED" in report.findings
    assert "REQUESTED_CAPITAL_ABOVE_CAPACITY" in report.findings
    assert report.minimum_capacity_usd == pytest.approx(50_000.0)


def test_rejects_excessive_implementation_cost() -> None:
    weights, adv, spreads, _ = evidence()
    spreads.iloc[1] = 75.0

    report = audit_execution_capacity(
        weights,
        adv,
        spreads,
        policy=policy(max_period_cost_bps=20.0),
    )

    assert not report.passed
    assert "PERIOD_COST_LIMIT_EXCEEDED" in report.findings
    assert report.maximum_cost_bps > 20.0


def test_borrow_cost_is_separate_and_can_trigger_rejection() -> None:
    weights, adv, spreads, borrow = evidence()
    borrow.loc[:, :] = 20_000.0

    report = audit_execution_capacity(
        weights,
        adv,
        spreads,
        annual_borrow_bps=borrow,
        policy=policy(max_period_cost_bps=30.0),
    )

    assert not report.passed
    assert report.periods[0].borrow_cost_bps == pytest.approx(39.6825396825)


def test_report_is_deterministic_bounded_and_json_ready() -> None:
    weights, adv, spreads, borrow = evidence(periods=8)
    active_policy = policy(max_reported_periods=3)

    first = audit_execution_capacity(weights, adv, spreads, annual_borrow_bps=borrow, policy=active_policy)
    second = audit_execution_capacity(weights, adv, spreads, annual_borrow_bps=borrow, policy=active_policy)

    assert first == second
    assert first.periods_truncated
    assert len(first.periods) == 3
    assert len(first.input_sha256) == 64
    assert len(first.evidence_sha256) == 64
    json.dumps(first.to_dict(), allow_nan=False)


def test_input_digest_changes_when_liquidity_evidence_changes() -> None:
    weights, adv, spreads, _ = evidence()
    first = audit_execution_capacity(weights, adv, spreads, policy=policy())
    adv.iloc[0, 0] += 1.0
    second = audit_execution_capacity(weights, adv, spreads, policy=policy())

    assert first.input_sha256 != second.input_sha256
    assert first.evidence_sha256 != second.evidence_sha256


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.set_axis(frame.index.tz_localize(None)), "timezone-aware"),
        (lambda frame: frame.iloc[::-1], "strictly increasing"),
        (lambda frame: frame.assign(**{"asset-0": float("nan")}), "finite"),
        (lambda frame: frame.rename(columns={"asset-0": " asset-0"}), "normalized"),
    ],
)
def test_malformed_weight_evidence_fails_closed(mutate, message: str) -> None:
    weights, adv, spreads, _ = evidence()

    with pytest.raises((TypeError, ValueError), match=message):
        audit_execution_capacity(mutate(weights), adv, spreads, policy=policy())


def test_misaligned_evidence_fails_closed() -> None:
    weights, adv, spreads, _ = evidence()

    with pytest.raises(ValueError, match="identical ordered assets"):
        audit_execution_capacity(weights, adv[adv.columns[::-1]], spreads, policy=policy())


@pytest.mark.parametrize(
    ("frame_name", "value", "message"),
    [
        ("adv", 0.0, "positive"),
        ("spread", -1.0, "non-negative"),
        ("borrow", -1.0, "non-negative"),
    ],
)
def test_invalid_market_evidence_fails_closed(frame_name: str, value: float, message: str) -> None:
    weights, adv, spreads, borrow = evidence()
    frames = {"adv": adv, "spread": spreads, "borrow": borrow}
    frames[frame_name].iloc[0, 0] = value

    with pytest.raises(ValueError, match=message):
        audit_execution_capacity(
            weights,
            adv,
            spreads,
            annual_borrow_bps=borrow,
            policy=policy(),
        )


def test_non_finite_derived_participation_fails_closed() -> None:
    weights, adv, spreads, _ = evidence()
    adv.iloc[0, 0] = float.fromhex("0x0.0000000000001p-1022")

    with pytest.raises(ValueError, match="derived participation"):
        audit_execution_capacity(weights, adv, spreads, policy=policy())


def test_leverage_neutrality_and_active_name_contracts_fail_closed() -> None:
    weights, adv, spreads, _ = evidence()

    with pytest.raises(ValueError, match="gross leverage"):
        audit_execution_capacity(weights * 2.0, adv, spreads, policy=policy())
    with pytest.raises(ValueError, match="dollar-neutral"):
        audit_execution_capacity(weights + 0.01, adv, spreads, policy=policy())
    with pytest.raises(ValueError, match="too few active"):
        audit_execution_capacity(
            weights,
            adv,
            spreads,
            policy=policy(min_active_assets=3),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"requested_capital_usd": float("nan")},
        {"max_participation_rate": 0.0},
        {"commission_bps": -1.0},
        {"max_period_cost_bps": 0.0},
        {"min_periods": True},
        {"min_assets": 5_001},
    ],
)
def test_invalid_policy_fails_closed(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(ExecutionAuditPolicy(), **changes)


def test_cli_has_distinct_policy_and_malformed_exit_codes(tmp_path, capsys) -> None:
    weights, adv, spreads, borrow = evidence()
    payload = {
        "timestamps": [timestamp.isoformat() for timestamp in weights.index],
        "assets": list(weights.columns),
        "target_weights": weights.values.tolist(),
        "adv_usd": adv.values.tolist(),
        "half_spread_bps": spreads.values.tolist(),
        "annual_borrow_bps": borrow.values.tolist(),
        "policy": asdict(policy(requested_capital_usd=100_000_000.0)),
    }
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert main([str(path), "--require-pass"]) == 2
    assert json.loads(capsys.readouterr().out)["passed"] is False
    payload["timestamps"] = [timestamp.replace("+00:00", "") for timestamp in payload["timestamps"]]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main([str(path)]) == 1
    assert "timezone" in json.loads(capsys.readouterr().out)["error"]
    path.write_text("{}", encoding="utf-8")
    assert main([str(path)]) == 1
    assert "error" in json.loads(capsys.readouterr().out)

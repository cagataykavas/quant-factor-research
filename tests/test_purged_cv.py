from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from quantfactor.purged_cv import (
    PurgedCVPolicy,
    audit_purged_walk_forward,
    load_artifact,
    main,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
START = datetime(2025, 1, 1, tzinfo=UTC)
NOW = datetime(2025, 8, 10, 12, 0, tzinfo=UTC)


def timestamp(value):
    return value.isoformat().replace("+00:00", "Z")


def observation(index, horizon_days=1):
    feature_at = START + timedelta(days=index)
    return {
        "observation_id": f"observation-secret-{index:04d}",
        "feature_at": timestamp(feature_at),
        "label_end_at": timestamp(feature_at + timedelta(days=horizon_days)),
    }


def observation_id(index):
    return f"observation-secret-{index:04d}"


def fold(fold_id, train_end, test_start, test_end):
    return {
        "fold_id": fold_id,
        "train_ids": [observation_id(index) for index in range(train_end + 1)],
        "test_ids": [
            observation_id(index) for index in range(test_start, test_end + 1)
        ],
    }


def artifact():
    return {
        "schema_version": "quant-purged-walk-forward/v1",
        "audit_id": "factor-research-secret",
        "dataset_sha256": HASH_A,
        "label_definition_sha256": HASH_B,
        "split_config_sha256": HASH_C,
        "created_at": "2025-08-10T11:50:00Z",
        "observations": [observation(index) for index in range(220)],
        "folds": [
            fold("fold-secret-1", 99, 101, 120),
            fold("fold-secret-2", 120, 122, 141),
            fold("fold-secret-3", 141, 143, 162),
        ],
    }


def codes(report):
    return {finding.code for finding in report.findings}


def test_valid_purged_expanding_folds_are_accepted_without_identity_leakage():
    report = audit_purged_walk_forward(artifact(), evaluated_at=NOW)

    assert report.accepted is True
    assert report.metrics == {
        "observation_count": 220,
        "fold_count": 3,
        "assignment_count": 423,
        "tested_observation_count": 60,
        "tested_observation_fraction": 60 / 220,
        "minimum_train_observations": 100,
        "minimum_test_observations": 20,
        "minimum_purge_gap_seconds": 86_400.0,
        "minimum_embargo_gap_seconds": 172_800.0,
        "maximum_label_horizon_seconds": 86_400.0,
        "artifact_age_seconds": 600.0,
    }
    encoded = json.dumps(report.as_dict())
    assert "secret" not in encoded
    assert len(report.policy_sha256) == 64


def test_purge_gap_catches_overlapping_forward_labels():
    value = artifact()
    value["observations"][99]["label_end_at"] = timestamp(START + timedelta(days=102))
    report = audit_purged_walk_forward(value, evaluated_at=NOW)

    assert report.accepted is False
    assert "PURGE_GAP_VIOLATION" in codes(report)
    assert report.metrics["minimum_purge_gap_seconds"] < 0


def test_positive_purge_and_embargo_policies_are_enforced_independently():
    purge = audit_purged_walk_forward(
        artifact(),
        evaluated_at=NOW,
        policy=PurgedCVPolicy(min_purge_gap_seconds=2 * 86_400),
    )
    assert "PURGE_GAP_VIOLATION" in codes(purge)

    embargo = audit_purged_walk_forward(
        artifact(),
        evaluated_at=NOW,
        policy=PurgedCVPolicy(min_embargo_gap_seconds=3 * 86_400),
    )
    assert "EMBARGO_GAP_VIOLATION" in codes(embargo)


def test_non_prefix_training_is_rejected():
    value = artifact()
    del value["folds"][0]["train_ids"][50]
    report = audit_purged_walk_forward(value, evaluated_at=NOW)
    assert "TRAIN_NOT_PREFIX" in codes(report)


def test_noncontiguous_test_block_is_rejected():
    value = artifact()
    value["folds"][0]["test_ids"].remove(observation_id(110))
    report = audit_purged_walk_forward(
        value,
        evaluated_at=NOW,
        policy=PurgedCVPolicy(min_test_observations=19),
    )
    assert "TEST_NOT_CONTIGUOUS" in codes(report)


def test_reused_and_backward_test_folds_are_rejected():
    reused = artifact()
    reused["folds"][1]["train_ids"].remove(observation_id(120))
    reused["folds"][1]["test_ids"][0] = observation_id(120)
    report = audit_purged_walk_forward(
        reused,
        evaluated_at=NOW,
        policy=PurgedCVPolicy(require_contiguous_test=False),
    )
    assert "TEST_OBSERVATION_REUSED" in codes(report)

    backward = artifact()
    backward["folds"][1]["test_ids"] = [
        observation_id(index) for index in range(115, 135)
    ]
    backward["folds"][1]["train_ids"] = [observation_id(index) for index in range(100)]
    report = audit_purged_walk_forward(backward, evaluated_at=NOW)
    assert "TEST_FOLDS_NOT_FORWARD" in codes(report)


def test_nonexpanding_training_window_is_rejected():
    value = artifact()
    value["folds"][1]["train_ids"] = value["folds"][0]["train_ids"][:]
    report = audit_purged_walk_forward(value, evaluated_at=NOW)
    assert "TRAIN_WINDOW_NOT_EXPANDING" in codes(report)


def test_minimum_fold_and_sample_policies_are_reported():
    value = artifact()
    value["folds"] = value["folds"][:2]
    report = audit_purged_walk_forward(value, evaluated_at=NOW)
    assert "INSUFFICIENT_FOLDS" in codes(report)

    small = artifact()
    small["folds"][0]["train_ids"] = small["folds"][0]["train_ids"][:99]
    small["folds"][0]["test_ids"] = small["folds"][0]["test_ids"][:19]
    report = audit_purged_walk_forward(small, evaluated_at=NOW)
    assert {
        "INSUFFICIENT_TRAIN_OBSERVATIONS",
        "INSUFFICIENT_TEST_OBSERVATIONS",
    } <= codes(report)


def test_policy_and_configuration_digests_bind_release_inputs():
    baseline = audit_purged_walk_forward(artifact(), evaluated_at=NOW)
    changed_policy = audit_purged_walk_forward(
        artifact(),
        evaluated_at=NOW,
        policy=PurgedCVPolicy(min_purge_gap_seconds=1.0),
    )
    changed_config = artifact()
    changed_config["split_config_sha256"] = HASH_A

    assert baseline.policy_sha256 != changed_policy.policy_sha256
    assert (
        baseline.configuration_sha256
        != audit_purged_walk_forward(
            changed_config, evaluated_at=NOW
        ).configuration_sha256
    )


def test_canonical_digest_ignores_input_order_and_timezone_spelling():
    value = artifact()
    reordered = deepcopy(value)
    reordered["observations"].reverse()
    reordered["folds"].reverse()
    for item in reordered["folds"]:
        item["train_ids"].reverse()
        item["test_ids"].reverse()
    reordered["created_at"] = "2025-08-10T14:50:00+03:00"

    assert (
        audit_purged_walk_forward(value, evaluated_at=NOW).artifact_sha256
        == audit_purged_walk_forward(reordered, evaluated_at=NOW).artifact_sha256
    )


def test_stale_and_future_artifacts_fail_closed():
    stale = artifact()
    stale["created_at"] = "2025-08-10T10:00:00Z"
    assert "STALE_ARTIFACT" in codes(
        audit_purged_walk_forward(
            stale,
            evaluated_at=NOW,
            policy=PurgedCVPolicy(max_artifact_age_seconds=60),
        )
    )

    future = artifact()
    future["created_at"] = "2025-08-10T12:06:00Z"
    assert "FUTURE_ARTIFACT" in codes(
        audit_purged_walk_forward(future, evaluated_at=NOW)
    )


def test_artifact_cannot_predate_label_completion():
    value = artifact()
    value["created_at"] = "2025-08-01T00:00:00Z"
    with pytest.raises(ValueError):
        audit_purged_walk_forward(value, evaluated_at=NOW)


@pytest.mark.parametrize(
    "payload",
    [b"", b"[]", b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff", b"{"],
)
def test_strict_loader_rejects_malformed_or_ambiguous_json(payload):
    with pytest.raises(ValueError):
        load_artifact(payload)


def test_strict_loader_enforces_byte_budget():
    payload = json.dumps(artifact()).encode()
    with pytest.raises(ValueError):
        load_artifact(payload, max_bytes=len(payload) - 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version="v2"),
        lambda value: value.update(extra=True),
        lambda value: value.update(dataset_sha256="ABC"),
        lambda value: value.update(audit_id="bad id"),
        lambda value: value.update(created_at="2025-08-10"),
        lambda value: value.update(observations=[]),
        lambda value: value.update(folds=[]),
        lambda value: value["observations"][0].update(extra=True),
        lambda value: value["observations"][0].update(
            label_end_at=value["observations"][0]["feature_at"]
        ),
        lambda value: value["folds"][0].update(extra=True),
    ],
)
def test_rejects_malformed_contract(mutate):
    value = artifact()
    mutate(value)
    with pytest.raises(ValueError):
        audit_purged_walk_forward(value, evaluated_at=NOW)


def test_rejects_duplicate_or_unknown_membership():
    duplicate_observation = artifact()
    duplicate_observation["observations"].append(
        deepcopy(duplicate_observation["observations"][0])
    )
    with pytest.raises(ValueError):
        audit_purged_walk_forward(duplicate_observation, evaluated_at=NOW)

    duplicate_assignment = artifact()
    duplicate_assignment["folds"][0]["train_ids"].append(observation_id(0))
    with pytest.raises(ValueError):
        audit_purged_walk_forward(duplicate_assignment, evaluated_at=NOW)

    overlap = artifact()
    overlap["folds"][0]["train_ids"].append(observation_id(101))
    with pytest.raises(ValueError):
        audit_purged_walk_forward(overlap, evaluated_at=NOW)

    unknown = artifact()
    unknown["folds"][0]["test_ids"][0] = "unknown-observation"
    with pytest.raises(ValueError):
        audit_purged_walk_forward(unknown, evaluated_at=NOW)


def test_duplicate_feature_timestamps_and_long_horizons_are_rejected():
    duplicate_time = artifact()
    duplicate_time["observations"][1]["feature_at"] = duplicate_time["observations"][0][
        "feature_at"
    ]
    with pytest.raises(ValueError):
        audit_purged_walk_forward(duplicate_time, evaluated_at=NOW)

    long_horizon = artifact()
    long_horizon["observations"][0]["label_end_at"] = timestamp(
        START + timedelta(days=40)
    )
    with pytest.raises(ValueError):
        audit_purged_walk_forward(long_horizon, evaluated_at=NOW)


def test_resource_budgets_fail_closed():
    with pytest.raises(ValueError):
        audit_purged_walk_forward(
            artifact(), evaluated_at=NOW, policy=PurgedCVPolicy(max_observations=219)
        )
    with pytest.raises(ValueError):
        audit_purged_walk_forward(
            artifact(), evaluated_at=NOW, policy=PurgedCVPolicy(max_folds=2)
        )
    with pytest.raises(ValueError):
        audit_purged_walk_forward(
            artifact(), evaluated_at=NOW, policy=PurgedCVPolicy(max_assignments=420)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_folds": 1},
        {"min_purge_gap_seconds": -1.0},
        {"max_label_horizon_seconds": float("nan")},
        {"max_observations": True},
        {"require_train_prefix": "yes"},
    ],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        audit_purged_walk_forward(
            artifact(), evaluated_at=NOW, policy=PurgedCVPolicy(**kwargs)
        )


def test_evaluated_at_must_be_timezone_aware():
    with pytest.raises(ValueError):
        audit_purged_walk_forward(artifact(), evaluated_at=NOW.replace(tzinfo=None))


def test_findings_are_bounded():
    value = artifact()
    value["folds"] = [
        {
            "fold_id": f"fold-{index}",
            "train_ids": [observation_id(0)],
            "test_ids": [observation_id((index % 100) + 100)],
        }
        for index in range(300)
    ]
    report = audit_purged_walk_forward(
        value,
        evaluated_at=NOW,
        policy=PurgedCVPolicy(max_folds=300, max_assignments=600),
    )
    assert len(report.findings) == 256
    assert report.findings[-1].code == "FINDINGS_TRUNCATED"


def test_cli_exposes_accept_reject_and_malformed_codes(tmp_path, capsys):
    source = tmp_path / "artifact.json"
    output = tmp_path / "report.json"
    current = artifact()
    current["created_at"] = datetime.now(UTC).isoformat()
    source.write_text(json.dumps(current), encoding="utf-8")
    assert main([str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["accepted"] is True

    current["folds"][0]["train_ids"] = current["folds"][0]["train_ids"][:-1]
    source.write_text(json.dumps(current), encoding="utf-8")
    assert main([str(source)]) == 2
    assert json.loads(capsys.readouterr().out)["accepted"] is False

    source.write_text('{"duplicate":1,"duplicate":2}', encoding="utf-8")
    assert main([str(source)]) == 3
    malformed = json.loads(capsys.readouterr().out)
    assert malformed["error_code"] == "MALFORMED_ARTIFACT"
    assert "duplicate" not in json.dumps(malformed)

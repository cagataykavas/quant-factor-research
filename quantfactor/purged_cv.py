"""Fail-closed audit for purged expanding-window cross-validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, NoReturn

SCHEMA_VERSION = "quant-purged-walk-forward/v1"
REPORT_VERSION = "quant-purged-walk-forward-report/v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "audit_id",
        "dataset_sha256",
        "label_definition_sha256",
        "split_config_sha256",
        "created_at",
        "observations",
        "folds",
    }
)
_OBSERVATION_FIELDS = frozenset({"observation_id", "feature_at", "label_end_at"})
_FOLD_FIELDS = frozenset({"fold_id", "train_ids", "test_ids"})
_MAX_FINDINGS = 256


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        _invalid(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        _invalid(f"{name} must be finite and in [{minimum}, {maximum}]")
    return number


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        _invalid(f"{name} must be a bounded identifier")
    return value


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _invalid(f"{name} must be a canonical lowercase SHA-256 digest")
    return value


def _timestamp(name: str, value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        _invalid(f"{name} must be a bounded RFC 3339 timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _invalid(f"{name} must include a timezone offset")
    return parsed.astimezone(UTC)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _invalid("JSON objects must not contain duplicate fields")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _invalid(f"non-finite JSON constant is not allowed: {value}")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PurgedCVPolicy:
    min_folds: int = 3
    min_train_observations: int = 100
    min_test_observations: int = 20
    min_purge_gap_seconds: float = 0.0
    min_embargo_gap_seconds: float = 0.0
    max_label_horizon_seconds: float = 31 * 86_400.0
    require_train_prefix: bool = True
    require_contiguous_test: bool = True
    require_disjoint_test_folds: bool = True
    require_expanding_train: bool = True
    max_artifact_age_seconds: float = 86_400.0
    max_future_skew_seconds: float = 300.0
    max_observations: int = 100_000
    max_folds: int = 1_000
    max_assignments: int = 1_000_000
    max_input_bytes: int = 4_194_304

    def validate(self) -> None:
        _integer("min_folds", self.min_folds, 2, 10_000)
        _integer("min_train_observations", self.min_train_observations, 1, 100_000_000)
        _integer("min_test_observations", self.min_test_observations, 1, 100_000_000)
        _number("min_purge_gap_seconds", self.min_purge_gap_seconds, 0.0, 31_536_000.0)
        _number(
            "min_embargo_gap_seconds", self.min_embargo_gap_seconds, 0.0, 31_536_000.0
        )
        _number(
            "max_label_horizon_seconds",
            self.max_label_horizon_seconds,
            0.001,
            315_360_000.0,
        )
        _number(
            "max_artifact_age_seconds",
            self.max_artifact_age_seconds,
            0.001,
            31_536_000.0,
        )
        _number("max_future_skew_seconds", self.max_future_skew_seconds, 0.0, 86_400.0)
        _integer("max_observations", self.max_observations, 1, 1_000_000)
        _integer("max_folds", self.max_folds, 2, 10_000)
        _integer("max_assignments", self.max_assignments, 1, 10_000_000)
        _integer("max_input_bytes", self.max_input_bytes, 1, 16_777_216)
        for name in (
            "require_train_prefix",
            "require_contiguous_test",
            "require_disjoint_test_folds",
            "require_expanding_train",
        ):
            if not isinstance(getattr(self, name), bool):
                _invalid(f"{name} must be boolean")


@dataclass(frozen=True)
class Finding:
    code: str
    evidence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "severity": "error", "evidence": self.evidence}


@dataclass(frozen=True)
class PurgedCVReport:
    accepted: bool
    artifact_sha256: str
    audit_sha256: str
    configuration_sha256: str
    policy_sha256: str
    evaluated_at: str
    metrics: dict[str, int | float]
    findings: tuple[Finding, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_VERSION,
            "accepted": self.accepted,
            "artifact_sha256": self.artifact_sha256,
            "audit_sha256": self.audit_sha256,
            "configuration_sha256": self.configuration_sha256,
            "policy_sha256": self.policy_sha256,
            "evaluated_at": self.evaluated_at,
            "metrics": self.metrics,
            "findings": [finding.as_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class _Observation:
    observation_id: str
    feature_at: datetime
    label_end_at: datetime


@dataclass(frozen=True)
class _Fold:
    fold_id: str
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


def load_artifact(payload: bytes, *, max_bytes: int = 4_194_304) -> dict[str, Any]:
    """Load strict, bounded split evidence."""
    limit = _integer("max_bytes", max_bytes, 1, 16_777_216)
    if not payload or len(payload) > limit:
        _invalid("artifact byte size is outside the configured budget")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        _invalid("artifact root must be an object")
    return value


def _parse_observation(value: object, policy: PurgedCVPolicy) -> _Observation:
    if not isinstance(value, dict) or set(value) != _OBSERVATION_FIELDS:
        _invalid("each observation must contain exactly the documented fields")
    feature_at = _timestamp("feature_at", value["feature_at"])
    label_end_at = _timestamp("label_end_at", value["label_end_at"])
    if label_end_at <= feature_at:
        _invalid("label_end_at must be after feature_at")
    horizon = (label_end_at - feature_at).total_seconds()
    if horizon > policy.max_label_horizon_seconds:
        _invalid("observation exceeds max_label_horizon_seconds")
    return _Observation(
        observation_id=_identifier("observation_id", value["observation_id"]),
        feature_at=feature_at,
        label_end_at=label_end_at,
    )


def _id_list(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        _invalid(f"{name} must be a non-empty array")
    result = tuple(_identifier("observation_id", item) for item in value)
    if len(set(result)) != len(result):
        _invalid(f"{name} must not contain duplicate observation IDs")
    return result


def _parse_fold(value: object) -> _Fold:
    if not isinstance(value, dict) or set(value) != _FOLD_FIELDS:
        _invalid("each fold must contain exactly the documented fields")
    train_ids = _id_list("train_ids", value["train_ids"])
    test_ids = _id_list("test_ids", value["test_ids"])
    if set(train_ids) & set(test_ids):
        _invalid("train_ids and test_ids must be disjoint within a fold")
    return _Fold(
        fold_id=_identifier("fold_id", value["fold_id"]),
        train_ids=train_ids,
        test_ids=test_ids,
    )


def _contiguous(indices: list[int]) -> bool:
    return indices == list(range(indices[0], indices[-1] + 1))


def audit_purged_walk_forward(
    document: dict[str, Any],
    *,
    evaluated_at: datetime,
    policy: PurgedCVPolicy | None = None,
) -> PurgedCVReport:
    """Audit temporal membership, purging and embargo for expanding folds."""
    active_policy = policy or PurgedCVPolicy()
    active_policy.validate()
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        _invalid("evaluated_at must be timezone-aware")
    now = evaluated_at.astimezone(UTC)
    if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
        _invalid("artifact must contain exactly the documented fields")
    if document["schema_version"] != SCHEMA_VERSION:
        _invalid("unsupported schema_version")

    audit_id = _identifier("audit_id", document["audit_id"])
    dataset_sha256 = _sha256("dataset_sha256", document["dataset_sha256"])
    label_sha256 = _sha256(
        "label_definition_sha256", document["label_definition_sha256"]
    )
    split_sha256 = _sha256("split_config_sha256", document["split_config_sha256"])
    created_at = _timestamp("created_at", document["created_at"])

    observation_values = document["observations"]
    if not isinstance(observation_values, list) or not observation_values:
        _invalid("observations must be a non-empty array")
    if len(observation_values) > active_policy.max_observations:
        _invalid("observations exceeds max_observations")
    observations = [
        _parse_observation(value, active_policy) for value in observation_values
    ]
    observation_ids = [item.observation_id for item in observations]
    if len(set(observation_ids)) != len(observation_ids):
        _invalid("observation_id values must be unique")
    observations.sort(key=lambda item: (item.feature_at, item.observation_id))
    if any(
        left.feature_at >= right.feature_at for left, right in pairwise(observations)
    ):
        _invalid("feature_at timestamps must be strictly increasing")
    if max(item.label_end_at for item in observations) > created_at:
        _invalid("artifact cannot predate label completion")
    observation_by_id = {item.observation_id: item for item in observations}
    index_by_id = {
        item.observation_id: index for index, item in enumerate(observations)
    }

    fold_values = document["folds"]
    if not isinstance(fold_values, list) or not fold_values:
        _invalid("folds must be a non-empty array")
    if len(fold_values) > active_policy.max_folds:
        _invalid("folds exceeds max_folds")
    folds = [_parse_fold(value) for value in fold_values]
    fold_ids = [fold.fold_id for fold in folds]
    if len(set(fold_ids)) != len(fold_ids):
        _invalid("fold_id values must be unique")
    assignment_count = sum(len(fold.train_ids) + len(fold.test_ids) for fold in folds)
    if assignment_count > active_policy.max_assignments:
        _invalid("fold memberships exceed max_assignments")
    for fold in folds:
        unknown = (set(fold.train_ids) | set(fold.test_ids)) - observation_by_id.keys()
        if unknown:
            _invalid("fold references an unknown observation_id")
    folds.sort(
        key=lambda fold: (
            min(index_by_id[item] for item in fold.test_ids),
            fold.fold_id,
        )
    )

    canonical_document = {
        "schema_version": SCHEMA_VERSION,
        "audit_id": audit_id,
        "dataset_sha256": dataset_sha256,
        "label_definition_sha256": label_sha256,
        "split_config_sha256": split_sha256,
        "created_at": _utc(created_at),
        "observations": [
            {
                "observation_id": item.observation_id,
                "feature_at": _utc(item.feature_at),
                "label_end_at": _utc(item.label_end_at),
            }
            for item in observations
        ],
        "folds": [
            {
                "fold_id": fold.fold_id,
                "train_ids": sorted(fold.train_ids, key=index_by_id.__getitem__),
                "test_ids": sorted(fold.test_ids, key=index_by_id.__getitem__),
            }
            for fold in folds
        ],
    }
    artifact_sha256 = _digest(
        json.dumps(canonical_document, sort_keys=True, separators=(",", ":"))
    )

    findings: list[Finding] = []
    age_seconds = (now - created_at).total_seconds()
    if age_seconds > active_policy.max_artifact_age_seconds:
        findings.append(
            Finding(
                "STALE_ARTIFACT",
                {
                    "age_seconds": age_seconds,
                    "limit": active_policy.max_artifact_age_seconds,
                },
            )
        )
    if age_seconds < -active_policy.max_future_skew_seconds:
        findings.append(
            Finding(
                "FUTURE_ARTIFACT",
                {
                    "future_skew_seconds": -age_seconds,
                    "limit": active_policy.max_future_skew_seconds,
                },
            )
        )
    if len(folds) < active_policy.min_folds:
        findings.append(
            Finding(
                "INSUFFICIENT_FOLDS",
                {"observed": len(folds), "required": active_policy.min_folds},
            )
        )

    tested_ids: set[str] = set()
    previous_train_ids: set[str] | None = None
    previous_train_max_index = -1
    previous_test_max_index = -1
    purge_gaps: list[float] = []
    embargo_gaps: list[float] = []
    train_sizes: list[int] = []
    test_sizes: list[int] = []
    for fold in folds:
        fold_ref = _digest(fold.fold_id)
        train_indices = sorted(index_by_id[item] for item in fold.train_ids)
        test_indices = sorted(index_by_id[item] for item in fold.test_ids)
        train_sizes.append(len(train_indices))
        test_sizes.append(len(test_indices))
        if len(train_indices) < active_policy.min_train_observations:
            findings.append(
                Finding(
                    "INSUFFICIENT_TRAIN_OBSERVATIONS",
                    {
                        "fold_sha256": fold_ref,
                        "observed": len(train_indices),
                        "required": active_policy.min_train_observations,
                    },
                )
            )
        if len(test_indices) < active_policy.min_test_observations:
            findings.append(
                Finding(
                    "INSUFFICIENT_TEST_OBSERVATIONS",
                    {
                        "fold_sha256": fold_ref,
                        "observed": len(test_indices),
                        "required": active_policy.min_test_observations,
                    },
                )
            )
        if active_policy.require_train_prefix and train_indices != list(
            range(train_indices[-1] + 1)
        ):
            findings.append(Finding("TRAIN_NOT_PREFIX", {"fold_sha256": fold_ref}))
        if active_policy.require_contiguous_test and not _contiguous(test_indices):
            findings.append(Finding("TEST_NOT_CONTIGUOUS", {"fold_sha256": fold_ref}))
        if train_indices[-1] >= test_indices[0]:
            findings.append(Finding("TRAIN_NOT_BEFORE_TEST", {"fold_sha256": fold_ref}))
            purge_gap = -1.0
            embargo_gap = -1.0
        else:
            test_start = observations[test_indices[0]].feature_at
            latest_train_label_end = max(
                observation_by_id[item].label_end_at for item in fold.train_ids
            )
            latest_train_feature = max(
                observation_by_id[item].feature_at for item in fold.train_ids
            )
            purge_gap = (test_start - latest_train_label_end).total_seconds()
            embargo_gap = (test_start - latest_train_feature).total_seconds()
            if purge_gap < active_policy.min_purge_gap_seconds:
                findings.append(
                    Finding(
                        "PURGE_GAP_VIOLATION",
                        {
                            "fold_sha256": fold_ref,
                            "observed_seconds": purge_gap,
                            "required_seconds": active_policy.min_purge_gap_seconds,
                        },
                    )
                )
            if embargo_gap < active_policy.min_embargo_gap_seconds:
                findings.append(
                    Finding(
                        "EMBARGO_GAP_VIOLATION",
                        {
                            "fold_sha256": fold_ref,
                            "observed_seconds": embargo_gap,
                            "required_seconds": active_policy.min_embargo_gap_seconds,
                        },
                    )
                )
        purge_gaps.append(purge_gap)
        embargo_gaps.append(embargo_gap)

        current_train_ids = set(fold.train_ids)
        current_test_ids = set(fold.test_ids)
        if active_policy.require_disjoint_test_folds and tested_ids & current_test_ids:
            findings.append(
                Finding("TEST_OBSERVATION_REUSED", {"fold_sha256": fold_ref})
            )
        if previous_test_max_index >= test_indices[0]:
            findings.append(
                Finding("TEST_FOLDS_NOT_FORWARD", {"fold_sha256": fold_ref})
            )
        if (
            active_policy.require_expanding_train
            and previous_train_ids is not None
            and (
                not previous_train_ids < current_train_ids
                or train_indices[-1] <= previous_train_max_index
            )
        ):
            findings.append(
                Finding("TRAIN_WINDOW_NOT_EXPANDING", {"fold_sha256": fold_ref})
            )
        tested_ids.update(current_test_ids)
        previous_train_ids = current_train_ids
        previous_train_max_index = train_indices[-1]
        previous_test_max_index = test_indices[-1]

    findings.sort(
        key=lambda item: (item.code, json.dumps(item.evidence, sort_keys=True))
    )
    finding_count = len(findings)
    if finding_count > _MAX_FINDINGS:
        findings = findings[: _MAX_FINDINGS - 1]
        findings.append(
            Finding(
                "FINDINGS_TRUNCATED",
                {"observed": finding_count, "reported": _MAX_FINDINGS},
            )
        )

    configuration = {
        "dataset_sha256": dataset_sha256,
        "label_definition_sha256": label_sha256,
        "split_config_sha256": split_sha256,
    }
    max_label_horizon = max(
        (item.label_end_at - item.feature_at).total_seconds() for item in observations
    )
    metrics: dict[str, int | float] = {
        "observation_count": len(observations),
        "fold_count": len(folds),
        "assignment_count": assignment_count,
        "tested_observation_count": len(tested_ids),
        "tested_observation_fraction": len(tested_ids) / len(observations),
        "minimum_train_observations": min(train_sizes),
        "minimum_test_observations": min(test_sizes),
        "minimum_purge_gap_seconds": min(purge_gaps),
        "minimum_embargo_gap_seconds": min(embargo_gaps),
        "maximum_label_horizon_seconds": max_label_horizon,
        "artifact_age_seconds": age_seconds,
    }
    return PurgedCVReport(
        accepted=not findings,
        artifact_sha256=artifact_sha256,
        audit_sha256=_digest(audit_id),
        configuration_sha256=_digest(
            json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        ),
        policy_sha256=_digest(
            json.dumps(asdict(active_policy), sort_keys=True, separators=(",", ":"))
        ),
        evaluated_at=_utc(now),
        metrics=metrics,
        findings=tuple(findings),
    )


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=4_194_304)
    args = parser.parse_args(argv)
    try:
        policy = PurgedCVPolicy(max_input_bytes=args.max_bytes)
        if args.artifact.stat().st_size > policy.max_input_bytes:
            _invalid("artifact byte size is outside the configured budget")
        document = load_artifact(
            args.artifact.read_bytes(), max_bytes=policy.max_input_bytes
        )
        report = audit_purged_walk_forward(
            document,
            evaluated_at=datetime.now(UTC),
            policy=policy,
        )
        payload = json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":"))
        exit_code = 0 if report.accepted else 2
    except (OSError, ValueError):
        payload = json.dumps(
            {
                "schema_version": REPORT_VERSION,
                "accepted": False,
                "error_code": "MALFORMED_ARTIFACT",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        exit_code = 3
    if args.output is None:
        print(payload)
    else:
        _write_atomic(args.output, payload)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

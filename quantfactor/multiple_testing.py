"""Multiple-testing controls for factor discovery pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Iterable


@dataclass(frozen=True)
class HypothesisResult:
    """Auditable result for one tested factor hypothesis."""

    name: str
    p_value: float
    adjusted_p_value: float
    rejected: bool

    def to_dict(self) -> dict[str, str | float | bool]:
        return asdict(self)


def benjamini_hochberg(
    hypotheses: Iterable[tuple[str, float]], *, false_discovery_rate: float = 0.05
) -> list[HypothesisResult]:
    """Control expected false discovery rate with Benjamini-Hochberg.

    Results preserve input order. Adjusted p-values are monotone across ranked
    hypotheses and can be compared with any reporting threshold; ``rejected``
    records the decision for ``false_discovery_rate``.
    """
    if not isfinite(false_discovery_rate) or not 0 < false_discovery_rate < 1:
        raise ValueError("false_discovery_rate must be finite and strictly between 0 and 1")

    records = list(hypotheses)
    if not records:
        raise ValueError("at least one hypothesis is required")

    seen: set[str] = set()
    for name, p_value in records:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("hypothesis names must be non-empty strings")
        if name in seen:
            raise ValueError(f"duplicate hypothesis name: {name}")
        seen.add(name)
        if not isinstance(p_value, (int, float)) or not isfinite(float(p_value)):
            raise ValueError(f"p-value for {name} must be finite")
        if not 0 <= p_value <= 1:
            raise ValueError(f"p-value for {name} must be between 0 and 1")

    ranked = sorted(enumerate(records), key=lambda item: (item[1][1], item[0]))
    count = len(ranked)

    adjusted_by_index: dict[int, float] = {}
    running_minimum = 1.0
    for reverse_rank, (original_index, (_, p_value)) in enumerate(reversed(ranked), start=1):
        rank = count - reverse_rank + 1
        running_minimum = min(running_minimum, float(p_value) * count / rank)
        adjusted_by_index[original_index] = min(running_minimum, 1.0)

    return [
        HypothesisResult(
            name=name,
            p_value=float(p_value),
            adjusted_p_value=adjusted_by_index[index],
            rejected=adjusted_by_index[index] <= false_discovery_rate,
        )
        for index, (name, p_value) in enumerate(records)
    ]

import pytest

from quantfactor.multiple_testing import benjamini_hochberg


def test_controls_false_discovery_and_preserves_input_order():
    results = benjamini_hochberg(
        [("quality", 0.04), ("momentum", 0.001), ("value", 0.01), ("noise", 0.50)],
        false_discovery_rate=0.05,
    )

    assert [result.name for result in results] == ["quality", "momentum", "value", "noise"]
    assert [result.adjusted_p_value for result in results] == pytest.approx(
        [0.0533333333, 0.004, 0.02, 0.5]
    )
    assert [result.rejected for result in results] == [False, True, True, False]


def test_adjusted_values_are_monotone_in_ranked_order():
    results = benjamini_hochberg([("a", 0.01), ("b", 0.011), ("c", 0.03)])
    adjusted = [result.adjusted_p_value for result in results]
    assert adjusted == sorted(adjusted)
    assert adjusted == pytest.approx([0.0165, 0.0165, 0.03])


@pytest.mark.parametrize(
    "hypotheses, message",
    [
        ([], "at least one"),
        ([("", 0.1)], "non-empty"),
        ([("same", 0.1), ("same", 0.2)], "duplicate"),
        ([("bad", -0.1)], "between 0 and 1"),
        ([("bad", float("nan"))], "finite"),
    ],
)
def test_rejects_invalid_hypothesis_sets(hypotheses, message):
    with pytest.raises(ValueError, match=message):
        benjamini_hochberg(hypotheses)


@pytest.mark.parametrize("rate", [0, 1, -0.1, float("inf"), float("nan")])
def test_rejects_invalid_false_discovery_rate(rate):
    with pytest.raises(ValueError, match="false_discovery_rate"):
        benjamini_hochberg([("factor", 0.01)], false_discovery_rate=rate)


def test_results_are_json_ready():
    result = benjamini_hochberg([("momentum", 0.01)])[0]
    assert result.to_dict() == {
        "name": "momentum",
        "p_value": 0.01,
        "adjusted_p_value": 0.01,
        "rejected": True,
    }

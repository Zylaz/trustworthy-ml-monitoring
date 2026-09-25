import math

import pytest

from trustworthy_ml_monitoring.validation import validate_features

@pytest.mark.parametrize("bad_value", [True, "2.0", None])
def test_invalid_value_type_raises_value_error(bad_value: object) -> None:
    with pytest.raises(ValueError):
        validate_features({"energy": bad_value}, ("energy",))


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_value_raises_value_error(bad_value: float) -> None:
    with pytest.raises(ValueError):
        validate_features({"energy": bad_value}, ("energy",))

def test_valid_record_returns_ordered_floats_without_mutating_input() -> None:
    record = {"energy": 2, "pitch": 1.5}

    result = validate_features(record, ("pitch", "energy"))

    assert list(result) == ["pitch", "energy"]
    assert result == {"pitch": 1.5, "energy": 2.0}
    assert all(type(value) is float for value in result.values())
    assert result is not record
    assert record == {"energy": 2, "pitch": 1.5}


def test_missing_feature_raises_value_error() -> None:
    with pytest.raises(ValueError):
        validate_features({"energy": 2}, ("energy", "pitch"))

def test_extra_feature_raises_value_error() -> None:
    with pytest.raises(ValueError):
        validate_features(
            {"energy": 2, "pitch": 1.5, "label": 1},
            ("energy", "pitch"),
        )
import math
from collections.abc import Mapping


def validate_features(
    record: Mapping[str, object],
    expected_names: tuple[str, ...],
) -> dict[str, float]:
    expected = set(expected_names)
    actual = set(record)

    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(f"Feature names differ; missing={missing}, extra={extra}")

    validated: dict[str, float] = {}

    for name in expected_names:
        value = record[name]

        if type(value) not in (int, float):
            raise ValueError(f"{name!r} must be a built-in int or float")

        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name!r} cannot be represented as a float") from exc

        if not math.isfinite(number):
            raise ValueError(f"{name!r} must be finite")

        validated[name] = number

    return validated
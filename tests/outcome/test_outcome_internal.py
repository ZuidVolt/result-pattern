# ruff: noqa: PLC2701

# mypy: ignore-errors

from result import Outcome


def test_outcome_has_error_logic() -> None:
    """Exercise complex has_error() branches."""
    # 1. None is never an error
    assert Outcome(1, None).has_error() is False

    # 2. Singular objects are errors
    assert Outcome(1, ValueError()).has_error() is True
    assert Outcome(1, "error string").has_error() is True
    assert Outcome(1, b"error bytes").has_error() is True

    # 3. Collections are errors ONLY if non-empty
    assert Outcome(1, [1]).has_error() is True
    assert Outcome(1, []).has_error() is False
    assert Outcome(1, {1}).has_error() is True
    assert Outcome(1, set()).has_error() is False
    assert Outcome(1, (1,)).has_error() is True
    assert Outcome(1, ()).has_error() is False


def test_outcome_repr() -> None:
    """Verify Outcome representation."""
    out = Outcome(10, "fail")
    assert repr(out) == "Outcome(value=10, error='fail')"

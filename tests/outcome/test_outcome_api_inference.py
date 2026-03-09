# ruff: noqa: ANN201, PLR2004

# mypy: disable-error-code="no-untyped-def, var-annotated"

from typing import Any

from result import is_ok
from result.outcome import Outcome, catch_outcome


def test_inference_outcome_mapping():
    """Verify type tracing through Outcome.map."""
    out = Outcome(10, "warn")
    # int -> str
    res = out.map(str)

    assert res.value == "10"
    # Checker should know res.value is str
    val: str = res.value
    assert val == "10"


def test_inference_outcome_to_result():
    """Verify type tracing during Outcome -> Result conversion."""
    out = Outcome(10, ValueError("fail"))
    res = out.to_result()

    if is_ok(res):
        # Checker should know res.ok() is int
        val: int = res.ok()
        assert val == 10


def test_inference_outcome_unpacking():
    """Verify type tracing through native unpacking."""

    def get_data() -> Outcome[int, str]:
        return Outcome(42, "slow")

    val, err = get_data()
    # Checker should know types
    val_check: int = val
    err_check: str | None = err
    assert val_check == 42
    assert err_check == "slow"


def test_inference_outcome_map_exc():
    """Verify inference for Outcome.map_exc."""
    out = Outcome(10, ValueError("fail"))
    res = out.map_exc({ValueError: "invalid"})
    # value is int, error is str
    val: int = res.value
    err: Any = res.error
    assert val == 10
    assert err == "invalid"


def test_inference_catch_outcome():
    """Verify inference for catch_outcome."""

    @catch_outcome({ValueError: "invalid"}, default=0)
    def risky(s: str) -> int:
        return int(s)

    res = risky("abc")
    # Should be Outcome[int, str | ValueError]
    val: int = res.value
    assert val == 0

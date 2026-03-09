from typing import Any

from result import Err, Ok, Result, is_ok


def test_precision_cast_and_chaining() -> None:
    """Verify complex union inference and manual overrides via cast_types."""

    def step1(x: int) -> Result[str, ValueError]:
        return Ok(str(x)) if x > 0 else Err(ValueError("fail"))

    def step2(s: str) -> Result[int, KeyError]:
        # ruff: noqa: PLR2004
        return Ok(len(s)) if len(s) < 5 else Err(KeyError("too long"))

    # 1. Complex Chaining with Union Error Types
    res = Ok(10).and_then(step1).and_then(step2)

    # 2. Using .unsafe.cast_types to widen or narrow
    forced: Result[Any, Exception] = res.unsafe.cast_types[Any, Exception]()
    assert is_ok(forced)

    # 3. Using it in a pipeline to resolve variance disputes
    def process_any(r: Result[Any, Exception]) -> str:
        # For the test, we know it's Ok
        return str(r.ok())

    output: str = process_any(res.unsafe.cast_types[Any, Exception]())
    assert output == "2"

    # 4. Deep Chaining
    def to_int(f: float) -> Result[int, Any]:
        return Ok[int](int(f))

    # basedpyright needs help with the chain after a cast
    casted: Result[float, Exception] = Ok(1).unsafe.cast_types[float, Exception]()
    final: Result[int, Any] = casted.and_then(to_int)
    assert final == Ok(1)

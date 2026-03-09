# ruff: noqa: ANN201, ANN202, ANN001, RUF029, B901, PLR2004, FBT003

# mypy: disable-error-code="no-untyped-def, no-untyped-call, var-annotated, arg-type, type-arg, misc"

# pyright: reportGeneralTypeIssues=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnusedVariable=false
# pyright: reportReturnType=false
# pyright: reportArgumentType=false

import operator
from typing import Any

import pytest

from result import (
    Do,
    DoAsync,
    Err,
    Ok,
    Result,
    any_ok,
    as_err,
    catch,
    catch_call,
    combine,
    do_notation,
    do_notation_async,
    is_err,
    is_ok,
    map2,
    partition,
)
from result.combinators import (
    flow,
    partition_exceptions,
    traverse,
    try_fold,
    validate,
)

# --- Section 1: Zero Annotations (Pure Gradual Inference) ---
# Testing how well the library infers types when the user provides NO hints.


def test_inference_chaining():
    """Trace types through long functional chains."""
    # int -> str -> list[str] -> int
    res = Ok(10).map(str).map(lambda s: [s]).and_then(lambda items: Ok(len(items)))

    assert res == Ok(1)
    if is_ok(res):
        val = res.ok()
        assert val == 1


def test_inference_product():
    """Verify that product correctly infers tuple types."""
    res1 = Ok(1)
    res2 = Ok("a")
    res3 = Ok(True)

    zipped = res1.product(res2).product(res3)
    assert zipped == Ok(((1, "a"), True))


def test_inference_combinators():
    """Verify inference for collection helpers."""
    results = [Ok(1), Ok(2), Ok(3)]
    combined = combine(results)
    assert combined == Ok([1, 2, 3])

    p_results = [Ok(1), Err("fail")]
    p_oks, p_errs = partition(p_results)
    val_ok = p_oks[0]
    val_err = p_errs[0]
    assert val_ok == 1
    assert val_err == "fail"


def test_inference_map2():
    """Verify map2 correctly infers the binary function types."""
    res = map2(Ok(10), Ok(20), operator.add)
    assert res == Ok(30)


def test_inference_any_ok():
    """Verify any_ok correctly infers the success type."""
    any_results = [Err("a"), Ok(100)]
    res = any_ok(any_results)
    assert res == Ok(100)


def test_inference_do_notation():
    @do_notation
    def workflow():
        a = yield Ok(10)
        b = yield Ok(a + 10)
        return b

    res = workflow()
    assert res == Ok(20)
    if is_ok(res):
        expected = 20
        assert res.ok() == expected


@pytest.mark.asyncio
async def test_inference_async_do_notation():
    @do_notation_async
    async def workflow_async():
        a = yield Ok(10)
        yield Ok(a + 5)

    res = await workflow_async()
    assert res == Ok(15)


def test_inference_transpose():
    res = Ok(10)
    transposed = res.transpose()
    assert transposed == Ok(10)


def test_inference_catch_decorator():
    @catch(ValueError)
    def parse(s):
        return int(s)

    assert parse("123") == Ok(123)
    assert is_err(parse("fail"))


# --- Section 2: Basic Function Annotations ---
# Testing ergonomics when users provide standard functional annotations.


def test_annotated_do_notation() -> None:
    @do_notation
    def workflow() -> Do[int, str]:
        # Variable types should be inferred from Ok()
        a = yield Ok(10)
        b = yield Ok(a + 10)
        val: int = b
        return val

    assert workflow() == Ok(20)


@pytest.mark.asyncio
async def test_annotated_async_do_notation() -> None:
    @do_notation_async
    async def workflow_async() -> DoAsync[int, str]:
        a = yield Ok(10)
        yield Ok(a + 5)

    assert await workflow_async() == Ok(15)


def test_annotated_catch() -> None:
    @catch(ValueError)
    def parse(s: str) -> int:
        return int(s)

    res = parse("123")
    assert res == Ok(123)


# --- Section 3: Advanced Inference & Detection ---


def test_erasure_catch_context_result():
    """Verify that CatchContext preserves types."""
    with catch(ValueError) as ctx:
        ctx.set(42)

    final_res = ctx.result
    assert final_res is not None
    if is_ok(final_res):
        val = final_res.ok()
        assert val == 42


def test_erasure_nested_flatten() -> None:
    """Verify type info after flattening."""
    # mypy needs a little help here due to the complex overload
    res: Result[int, Any] = Ok(Ok(10)).flatten()
    assert res == Ok(10)
    if is_ok(res):
        val = res.ok()
        assert val == 10


def test_inference_mixed_combinators():
    """Verify that checkers correctly infer union error types."""

    class ErrorAError(Exception):
        pass

    class ErrorBError(Exception):
        pass

    results = [Ok(1), Err(ErrorAError()), Err(ErrorBError())]
    res = combine(results)
    assert is_err(res)


def test_inference_complex_do_loop():
    """Verify inference in do_notation with loops."""

    @do_notation
    def sum_even_numbers(inputs):
        total = 0
        for i in inputs:
            if i % 2 == 0:
                val = yield Ok(i)
                total += val
        return total

    res = sum_even_numbers([1, 2, 3, 4])
    assert res == Ok(6)


def test_inference_catch_call() -> None:
    """Verify inference for inline catch_call."""
    res = catch_call(ValueError, int, "123")
    assert res == Ok(123)
    if is_ok(res):
        # Checker should trace val as int
        val = res.ok()
        assert val == 123

    # Tuple of exceptions inference
    res2 = catch_call((ValueError, TypeError), int, "123")
    if is_err(res2):
        err = res2.err()
        assert isinstance(err, ValueError | TypeError)


def test_inference_as_err() -> None:
    """Verify type tracing for as_err."""
    e = ValueError("fail")
    res = as_err(e, {ValueError: "invalid"})
    if is_err(res):
        # Checker should know it's str | ValueError
        val: str | ValueError = res.err()
        assert val == "invalid"


def test_inference_result_map_exc() -> None:
    """Verify inference for Result.map_exc."""
    res = Err(ValueError("fail")).map_exc({ValueError: "invalid"})
    if is_err(res):
        # Should know it is str
        val: Any = res.err()
        assert val == "invalid"


def test_inference_catch_mapping():
    """Verify inference for catch with mapping."""

    @catch({ValueError: "invalid"})
    def risky(s: str) -> int:
        return int(s)

    res = risky("abc")
    if is_err(res):
        val: Any = res.err()
        assert val == "invalid"


def test_inference_partition_exceptions():
    """Verify type tracing through partition_exceptions."""
    items = [1, ValueError("a")]
    # Should be (list[Ok[int]], list[Err[ValueError]])
    oks, errs = partition_exceptions(items)

    val = oks[0].ok()
    assert val == 1

    err = errs[0].err()
    assert isinstance(err, ValueError)


# --- Section 4: Combinator Inference ---


def test_inference_traverse():
    """Verify type tracing through traverse."""
    items = [1, 2, 3]
    # int -> Result[str, Any]
    res = traverse(items, lambda x: Ok(str(x)))
    assert res == Ok(["1", "2", "3"])
    if is_ok(res):
        # Should know it is list[str]
        val = res.ok()
        assert val[0] == "1"


def test_inference_try_fold():
    """Verify type tracing through try_fold."""
    items = [1, 2, 3]
    # (int, int) -> Result[int, Any]
    res = try_fold(items, 0, lambda acc, x: Ok(acc + x))
    assert res == Ok(6)


def test_inference_validate_overload():
    """Verify tuple preservation in validate overloads."""
    r1 = Ok(1)
    r2 = Ok("a")
    res = validate(r1, r2)
    assert res == Ok((1, "a"))
    if is_ok(res):
        val = res.ok()
        # Should be able to unpack specifically
        v1, v2 = val
        assert v1 == 1
        assert v2 == "a"


def test_inference_flow_overload():
    """Verify type preservation in flow overloads."""
    # str -> int -> list[int]
    res = flow("10", lambda s: Ok(int(s)), lambda i: Ok([i]))
    assert res == Ok([10])


def test_inference_cast_types():
    """Verify manual type override via cast_types."""
    # Force a broader error type
    res: Result[int, ValueError] = Ok(10)
    res.unsafe.cast_types[int, Exception]()

    # Force a different success type (unsafe, but allowed by cast)
    overridden = res.unsafe.cast_types[str, ValueError]()
    if is_ok(overridden):
        # We forced str but the runtime value is still 10
        val: Any = overridden.ok()
        assert val == 10

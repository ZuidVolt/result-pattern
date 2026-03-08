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

import pytest

from result import (
    Do,
    DoAsync,
    Err,
    Ok,
    any_ok,
    combine,
    do_notation,
    do_notation_async,
    is_err,
    is_ok,
    map2,
    partition,
    safe,
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

    p_oks, p_errs = partition([Ok(1), Err("fail")])
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
    res = any_ok([Err("a"), Ok(100)])
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
        assert res.ok() == 20


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


def test_inference_safe_decorator():
    @safe(ValueError)
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


def test_annotated_safe() -> None:
    @safe(ValueError)
    def parse(s: str) -> int:
        return int(s)

    res = parse("123")
    assert res == Ok(123)


# --- Section 3: Advanced Inference & Detection ---


def test_erasure_safe_context_result():
    """Verify that SafeContext preserves types."""
    with safe(ValueError) as ctx:
        ctx.set(42)

    final_res = ctx.result
    assert final_res is not None
    if is_ok(final_res):
        val = final_res.ok()
        assert val == 42


def test_erasure_nested_flatten():
    """Verify type info after flattening."""
    res = Ok(Ok(10)).flatten()
    assert res == Ok(10)
    if is_ok(res):
        # res.ok() should ideally not be Any here if possible
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

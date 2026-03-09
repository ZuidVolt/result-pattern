# ruff: noqa: RUF029, PLC2701


# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportPrivateUsage=false
# pyright: reportUnusedVariable=false
# pyright: reportUnnecessaryTypeIgnoreComment=false
# pyright: reportUnnecessaryComparison=false
# pyright: reportReturnType=false
# pyright: reportImplicitOverride=false
# pyright: reportMissingSuperCall=false
# pyright: reportUnusedParameter=false
# pyright: reportUnusedCallResult=false
# pyright: reportUnreachable=false
# pyright: reportInvalidCast=false
# pyright: reportDeprecated=false
# pyright: reportAny=false

import operator
from typing import TYPE_CHECKING, Any, Never

import pytest
from hypothesis import given
from hypothesis import strategies as st

from result import (
    Do,
    DoAsync,
    Err,
    Ok,
    Result,
    UnwrapError,
    catch,
    catch_call,
    do_async,
    do_notation,
    do_notation_async,
    is_err,
)
from result.combinators import (
    flow,
    partition_exceptions,
    traverse,
    validate,
)
from result.result import _DoError, _raise_api_error

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# --- API Safeguards & Safety ---


@pytest.mark.asyncio
@given(st.integers())
async def test_exhaustive_variant_behavior_property(val: int) -> None:
    """Exhaustively verify variant-specific methods and unsafe proxies."""
    res_ok = Ok(val)
    res_err = Err(val)

    # 1. Functional Async (Direct calls for coverage)
    async def dummy(x: Any) -> Any:
        return x

    assert await res_ok.map_async(dummy) == Ok(val)
    assert await res_err.map_async(dummy) == res_err

    # 2. Unsafe Proxies
    assert res_ok.unsafe.unwrap() == val
    assert res_ok.unsafe.expect("msg") == val
    assert res_ok.unsafe.unwrap_or_raise(ValueError) == val
    with pytest.raises(UnwrapError):
        res_ok.unsafe.unwrap_err()
    with pytest.raises(UnwrapError):
        res_ok.unsafe.expect_err("msg")

    assert res_err.unsafe.unwrap_err() == val
    assert res_err.unsafe.expect_err("msg") == val
    with pytest.raises(UnwrapError):
        res_err.unsafe.unwrap()
    with pytest.raises(UnwrapError):
        res_err.unsafe.expect("msg")
    with pytest.raises(ValueError):
        res_err.unsafe.unwrap_or_raise(ValueError)


def test_educational_safeguards() -> None:
    """Verify that common API mistakes trigger educational guidance."""
    res_ok = Ok(10)
    res_err = Err("fail")

    for variant in [res_ok, res_err]:
        # Direct property access disabled
        with pytest.raises(AttributeError, match=r"Direct access to '.value'"):
            _ = variant.value

        # Crashing operations isolated in .unsafe
        for method in ["unwrap", "unwrap_err", "expect", "expect_err", "unwrap_or_raise"]:
            with pytest.raises(AttributeError, match=r"isolated in the '.unsafe' namespace"):
                m = getattr(variant, method)
                if method in {"unwrap", "unwrap_err"}:
                    m()
                else:
                    m("msg")

        # Legacy naming hints
        for method in ["inspect", "inspect_async", "inspect_err"]:
            with pytest.raises(AttributeError, match=r"renamed to '.*tap"):
                getattr(variant, method)(print)


# --- Coverage Mop-up (Cold Smell Tests) ---


@pytest.mark.asyncio
async def test_exhaustive_mop_up() -> None:  # noqa: PLR0915
    """Coverage Mop-up: Exercise unreachable or difficult-to-hit branches.

    These tests serve as a 'cold-smell' safety net, ensuring that every edge case
    in result.py is exercised to maintain 100% confidence in the implementation.
    """
    magic_val = 123
    res_ok: Ok[int] = Ok(magic_val)
    res_err: Err[str] = Err("error")

    # 1. Unsafe Proxies
    assert res_ok.unsafe.expect("msg") == magic_val
    assert res_ok.unsafe.unwrap_or_raise(ValueError) == magic_val
    with pytest.raises(UnwrapError):
        res_ok.unsafe.expect_err("msg")
    assert res_err.unsafe.expect_err("msg") == "error"

    # 2. Async Short-circuits
    async def dummy(x: Any) -> Any:
        return x

    assert await res_err.map_async(dummy) == Err("error")
    assert await res_err.tap_async(dummy) == Err("error")
    assert await res_err.and_then_async(lambda x: dummy(Ok(x))) == Err("error")

    # 3. Empty Generators
    async def empty_gen() -> AsyncGenerator[Result[int, str], Any]:
        if False:
            yield Ok(1)

    with pytest.raises(UnwrapError, match="ended without yielding"):
        await do_async(empty_gen())

    # 4. Generator Flow Invariants
    with pytest.raises(_DoError):
        next(iter(res_err))

    aiter_err = aiter(res_err)
    with pytest.raises(_DoError):
        await anext(aiter_err)

    # 5. Root-level Redirects (hitting _raise_api_error lines)
    for method in [
        "unwrap",
        "unwrap_err",
        "expect",
        "expect_err",
        "unwrap_or_raise",
        "inspect",
        "inspect_async",
        "inspect_err",
    ]:
        with pytest.raises(AttributeError, match=r"API Warning"):
            m = getattr(res_ok, method)
            if method in {"unwrap", "unwrap_err"}:
                m()
            else:
                m((lambda x, m_local=method: x if "inspect" in m_local else "msg"))

        with pytest.raises(AttributeError, match=r"API Warning"):
            m = getattr(res_err, method)
            if method in {"unwrap", "unwrap_err"}:
                m()
            else:
                # Bind method variable to avoid B023
                m((lambda x, m_local=method: x if "inspect" in m_local else "msg"))

    # 6. Err variant trivial methods
    assert res_err.ok() is None
    assert res_err.err() == "error"
    assert res_err.flatten() == res_err
    assert res_err.filter(lambda _: True, "fail") == res_err
    assert res_err.match(on_ok=lambda x: x, on_err=lambda e: e) == "error"

    # 7. Decorator exception paths
    @do_notation(catch=ValueError)
    def fail_sync() -> Do[Any, Any]:
        msg = "boom"
        raise ValueError(msg)
        yield Ok(1)

    assert is_err(fail_sync())

    @do_notation_async(catch=ValueError)
    async def fail_async() -> DoAsync[Any, Any]:
        msg = "boom async"
        raise ValueError(msg)
        yield Ok(1)

    assert is_err(await fail_async())

    @catch(ValueError)
    async def catch_async_fail() -> Never:
        msg = "safe boom"
        raise ValueError(msg)

    assert is_err(await catch_async_fail())


def test_api_error_default() -> None:
    """Hit the default case in _raise_api_error for unsupported methods."""
    with pytest.raises(AttributeError, match="not part of the supported Result API"):
        _raise_api_error("nonexistent_method")


def test_unwrap_error_chaining() -> None:
    """Verify that exceptions wrapped in Err are chained during panic."""
    cause = ValueError("original cause")
    res_err = Err(cause)

    with pytest.raises(UnwrapError) as exc_info:
        res_err.unsafe.unwrap()

    assert exc_info.value.__cause__ is cause

    with pytest.raises(UnwrapError) as exc_info:
        res_err.unsafe.expect("custom msg")

    assert exc_info.value.__cause__ is cause


def test_pattern_matching_ergonomics() -> None:
    """Verify __match_args__ alignment for cleaner matching."""
    val = 10
    err_msg = "fail"

    def check_match(res: Result[int, str]) -> None:
        match res:
            case Ok(value):
                assert value == val
            case Err(error):
                assert error == err_msg

    check_match(Ok(val))
    check_match(Err(err_msg))


@pytest.mark.asyncio
async def test_async_remapping_mop_up() -> None:
    """Exercise remapping in async do blocks."""

    class InternalError(Exception):
        pass

    class DomainError(Exception):
        def __init__(self, cause: Exception) -> None:
            self.cause = cause

    @do_notation_async(remap={InternalError: DomainError})
    async def workflow_async() -> DoAsync[Any, Any]:
        # Result objects are not awaitable, only yielded in async do blocks
        err_res: Result[int, DomainError | InternalError] = Err(InternalError("fail"))
        yield err_res
        yield Ok(1)

    res = await workflow_async()
    assert is_err(res)
    assert isinstance(res.err(), DomainError)


def test_catch_context_unhandled_exception() -> None:
    """Verify catch context manager does not swallow unlisted exceptions."""
    with pytest.raises(KeyError), catch(ValueError) as _:
        raise KeyError("not caught")


# --- Combinators Internal Mop-up ---


def test_validate_internal_logic() -> None:
    """Exercise complex validation error paths."""
    res = validate(Ok(1), Err("e1"), Ok(2), Err("e2"))
    assert res == Err(["e1", "e2"])


def test_traverse_internal_logic() -> None:
    """Exercise traverse early exit."""
    calls = 0
    limit = 2

    def work(x: int) -> Result[int, str]:
        nonlocal calls
        calls += 1
        return Ok(x) if x < limit else Err("too big")

    res = traverse([1, 2, 3], work)
    assert res == Err("too big")
    assert calls == limit  # Proves early exit after '2'


def test_flow_internal_logic() -> None:
    """Exercise flow early exit."""
    calls = 0

    def f1(_x: int) -> Result[int, str]:
        nonlocal calls
        calls += 1
        return Err("stop")

    def f2(x: int) -> Result[int, str]:
        nonlocal calls
        calls += 1
        return Ok(x)

    res = flow(1, f1, f2)
    assert res == Err("stop")
    assert calls == 1  # f2 never called


def test_catch_call_internal_logic() -> None:
    """Exercise catch_call with various arguments."""

    def add_kwargs(a: int, b: int = 0) -> int:  # noqa: FURB118
        return a + b

    assert catch_call(ValueError, operator.add, 1, 2) == Ok(3)
    assert catch_call(ValueError, add_kwargs, 1, b=2) == Ok(3)


def test_partition_exceptions_internal_logic() -> None:
    """Exercise partition_exceptions with mixed inputs."""
    items = ["a", ValueError("err")]
    oks, errs = partition_exceptions(items)
    assert len(oks) == 1
    assert len(errs) == 1
    assert oks[0].ok() == "a"
    assert isinstance(errs[0].err(), ValueError)


def test_traceback_trimming() -> None:
    """Verify library frames use industry-standard markers for hiding."""

    @catch(ValueError)
    def crash() -> Never:
        raise RuntimeError("boom")

    try:
        crash()
    except RuntimeError as e:
        # 1. Verify noise suppression (raise from None)
        assert e.__context__ is None

        # 2. Verify marker presence in the wrapper frame
        # We need to find the frame that belongs to the library's wrapper
        tb = e.__traceback__
        assert tb is not None

        # In this test environment, the chain is: test_func -> wrapper -> crash
        # wrapper should have the marker
        curr = tb
        found_marker = False
        while curr:
            if curr.tb_frame.f_locals.get("__tracebackhide__"):
                found_marker = True
                break
            curr = curr.tb_next

        assert found_marker, "Did not find __tracebackhide__ marker in the traceback chain"


@pytest.mark.asyncio
async def test_instructional_async_error() -> None:
    """Verify helpful message when async do block lacks a yield."""

    @do_notation_async
    async def empty_workflow() -> DoAsync[Any, Any]:
        if False:
            yield Ok(1)

    with pytest.raises(UnwrapError, match="Note: Async generators cannot use 'return value'"):
        await empty_workflow()

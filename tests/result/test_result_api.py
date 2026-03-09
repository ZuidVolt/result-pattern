# ruff: noqa: RUF029, B901

import json
import operator
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Never, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from result import (
    CatchContext,
    Do,
    DoAsync,
    Err,
    Ok,
    OkErr,
    Result,
    any_ok,
    as_err,
    catch,
    catch_call,
    combine,
    do,
    do_async,
    do_notation,
    do_notation_async,
    from_optional,
    is_err,
    is_ok,
    map2,
    partition,
)
from result.combinators import (
    add_context,
    ensure,
    flow,
    partition_exceptions,
    succeeds,
    traverse,
    try_fold,
    validate,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

# --- Core Functional Invariants (Algebraic Laws) ---


@given(st.integers())
def test_functor_identity_law(val: int) -> None:
    """Functor Identity: res.map(id) == res.

    Verifies that mapping the identity function over a Result (Ok or Err)
    does not change the Result.
    """
    res_ok = Ok(val)
    assert res_ok.map(lambda x: x) == res_ok
    assert "Ok(" in repr(res_ok)

    res_err = Err(val)
    assert res_err.map(lambda x: x) == res_err
    assert "Err(" in repr(res_err)


@given(st.integers())
def test_functor_composition_law(val: int) -> None:
    """Functor Composition: res.map(f).map(g) == res.map(lambda x: g(f(x))).

    Verifies that chaining multiple maps is equivalent to a single map
    of the composed functions.
    """

    def f(x: int) -> int:
        return x + 1

    def g(x: int) -> int:
        return x * 2

    res = Ok(val)
    assert res.map(f).map(g) == res.map(lambda x: g(f(x)))


@given(st.integers())
def test_monad_left_identity_law(val: int) -> None:
    """Monad Left Identity: Ok(val).and_then(f) == f(val).

    Verifies that wrapping a value in Ok and then binding a function
    is the same as applying the function to the value.
    """

    def f(x: int) -> Result[int, Never]:
        return Ok(x + 1)

    assert Ok(val).and_then(f) == f(val)


@given(st.integers())
def test_monad_right_identity_law(val: int) -> None:
    """Monad Right Identity: res.and_then(Ok) == res.

    Verifies that binding Ok (the 'return' function) to a Result
    does not change the Result.
    """
    res = Ok(val)
    assert res.and_then(Ok) == res


@given(st.integers())
def test_style_equivalence_property(val: int) -> None:
    """Equivalence: Method Chaining (map/and_then) vs @do_notation.

    Verifies that the imperative-style @do_notation syntax produces
    identical results to the functional-style method chaining.
    """

    def f(x: int) -> int:
        return x + 1

    def g(x: int) -> Result[int, Never]:
        return Ok(x * 2)

    # Chaining style
    chained = Ok(val).map(f).and_then(g)
    assert chained.is_ok() is True
    assert chained.is_err() is False

    # Do-notation style
    @do_notation()
    def procedural(x: int) -> Do[int, Any]:
        a = yield Ok(x)
        b = yield Ok(f(a))
        res_val: int = yield g(b)
        return res_val

    assert chained == procedural(val)


# --- Complex Integration Tests ---


def test_data_pipeline_integration() -> None:
    """Integration: Combine @safe, combine, and @do_notation in a realistic pipeline.

    This test simulates a multi-step data processing flow where external
    exceptions are lifted into Results, combined into a collection, and
    processed imperatively.
    """

    @catch(ValueError)
    def parse_int(s: str) -> int:
        return int(s)

    @do_notation()
    def sum_raw_inputs(raw_inputs: list[str]) -> Do[Any, Any]:
        # 1. Lift exceptions into Result containers
        results: list[Result[int, ValueError]] = [parse_int(i) for i in raw_inputs]

        # 2. Transpose list of Results into Result of list (all-or-nothing)
        ints: list[int] = yield combine(results)

        # 3. Sum the result
        return sum(ints)

    assert sum_raw_inputs(["1", "2", "3"]) == Ok(6)

    res_err: Result[int, Exception] = sum_raw_inputs(["1", "not a number", "3"])
    assert is_err(res_err)


@pytest.mark.asyncio
async def test_async_integration_flow() -> None:
    """Integration: Test complex async flows with do_notation_async and catch wrappers."""

    @catch(RuntimeError)
    async def fetch_user_name(user_id: int) -> str:

        if user_id < 0:
            msg = "invalid id"
            raise RuntimeError(msg)
        return f"User_{user_id}"

    @do_notation_async
    async def get_welcome_message(user_id: int) -> DoAsync[str, Exception]:
        name_res: Result[str, Exception] = await cast("Awaitable[Result[str, Exception]]", fetch_user_name(user_id))
        name: str = yield name_res
        yield Ok(f"Welcome, {name}!")

    assert await get_welcome_message(1) == Ok("Welcome, User_1!")
    res_err: Result[str, Exception] = await get_welcome_message(-1)
    assert is_err(res_err)


# --- Property-Based Coverage ---


@given(st.one_of(st.integers(), st.none()), st.text())
def test_from_optional_invariant(val: int | None, err: str) -> None:
    """Invariant: from_optional produces Ok(val) if not None, else Err(err)."""
    res = from_optional(val, err)
    if val is None:
        assert is_err(res)
        res_ok_val = res.ok()
        assert res_ok_val is None
        assert res.err() == err
    else:
        assert is_ok(res)
        assert res.ok() == val
        res_err_val = res.err()
        assert res_err_val is None


@given(st.integers(), st.text())
def test_mapping_variants_property(val: int, default: str) -> None:
    """Verify map_* and is_*_and variants behave correctly across variants."""
    f = str
    res_ok = Ok(val)
    res_err = Err(val)

    # Core mapping
    assert res_ok.map_or(default, f) == f(val)
    assert res_err.map_or(default, f) == default
    assert res_ok.map_or_else(lambda: default, f) == f(val)
    assert res_err.map_or_else(lambda: default, f) == default

    # Predicates
    assert res_ok.is_ok_and(lambda x: x == val) is True
    assert res_ok.is_err_and(lambda _: True) is False
    assert res_err.is_err_and(lambda e: e == val) is True
    assert res_err.is_ok_and(lambda _: True) is False


@given(st.integers(), st.text())
def test_side_effects_and_replacements_property(val: int, new_val: str) -> None:
    """Verify tap and replace methods across variants."""
    side_effects: list[int | str] = []
    res_ok = Ok(val)
    res_err = Err(new_val)

    # Success path side effects
    assert res_ok.tap(side_effects.append) == res_ok
    assert side_effects == [val]
    assert res_ok.tap_err(lambda _: side_effects.append("fail")) == res_ok
    assert side_effects == [val]

    # Error path side effects
    assert res_err.tap_err(side_effects.append) == res_err
    assert side_effects == [val, new_val]
    assert res_err.tap(lambda _: side_effects.append("fail")) == res_err
    assert side_effects == [val, new_val]

    # Replacements
    assert res_ok.replace(new_val) == Ok(new_val)
    assert res_ok.replace_err(val) == res_ok
    assert res_err.replace_err(val) == Err(val)
    assert res_err.replace(new_val) == res_err


@given(st.lists(st.booleans()))
def test_partition_property(bools: list[bool]) -> None:
    """Verify partitioning logic always correctly divides Results into collections."""
    results = [Ok(i) if b else Err(f"err_{i}") for i, b in enumerate(bools)]
    oks, errs = partition(results)
    assert len(oks) == sum(bools)
    assert len(errs) == len(bools) - sum(bools)


# --- Core Do-Notation Utilities ---


def test_do_inline() -> None:
    """Invariant: do() helper correctly unwraps sync generator expressions."""
    assert do(Ok(x + y) for x in Ok(1) for y in Ok(2)) == Ok(3)
    assert do(Ok(x + y) for x in Ok(1) for y in Err("fail")) == Err("fail")


@pytest.mark.asyncio
async def test_do_inline_async() -> None:
    """Invariant: do_async() helper correctly unwraps async generator expressions."""
    res: Result[int, Any] = await do_async(Ok(x + 2) async for x in Ok(1))
    assert res == Ok(3)


def test_do_notation_catch_basic() -> None:
    """Verify @do_notation catch parameter lifts exceptions into Err."""

    @do_notation(catch=ValueError)
    def risky(s: str) -> Do[int, Exception]:
        val: int = yield Ok(int(s))
        return val

    assert risky("10") == Ok(10)
    res: Result[int, Exception] = risky("not a number")
    assert is_err(res)


@pytest.mark.asyncio
async def test_do_notation_async_catch_basic() -> None:
    """Verify @do_notation_async catch parameter lifts exceptions into Err."""

    @do_notation_async(catch=ValueError)
    async def risky_async(s: str) -> DoAsync[int, Exception]:
        val: int = yield Ok(int(s))
        yield Ok(val)

    assert await risky_async("10") == Ok(10)
    res: Result[int, Exception] = await risky_async("not a number")
    assert is_err(res)


def test_covariance_support() -> None:
    """Verify true covariance support (Dog -> Animal)."""

    class Animal:
        pass

    class Dog(Animal):
        pass

    # Ok[Dog] should be assignable to Result[Animal, Any]
    res: Result[Animal, Any] = Ok(Dog())
    assert is_ok(res)


def test_convenience_utils() -> None:
    """Verify OkErr constant and fluent or_else on Ok."""
    assert isinstance(Ok(1), OkErr)
    assert isinstance(Err(1), OkErr)
    assert not isinstance(1, OkErr)

    # Use explicit lambda type to satisfy basedpyright in public API test
    res_ok: Ok[int] = Ok(10)

    def recover(_: object) -> Ok[int]:
        return Ok(0)

    assert res_ok.or_else(recover) == Ok(10)


def test_dunder_methods() -> None:
    """Verify __bool__, __hash__, and __add__ on Result variants."""
    ok1 = Ok(10)
    ok2 = Ok(5)
    err1 = Err("fail")
    err2 = Err("fatal")

    # __bool__
    assert bool(ok1) is True
    assert bool(err1) is False

    # __hash__
    assert hash(Ok(1)) != hash(Err(1))
    # Should be usable in sets/dicts
    d = {ok1: "a", err1: "b"}
    assert d[Ok(10)] == "a"
    assert d[Err("fail")] == "b"

    # __add__
    assert ok1 + ok2 == Ok(15)
    assert ok1 + err1 == err1
    assert err1 + ok1 == err1
    assert err1 + err2 == err1


def test_pattern_matching_type_narrowing() -> None:
    """Verify that pattern matching correctly narrows types for static analysis.

    This ensures that __match_args__ are correctly aligned with private
    field names, allowing tools like basedpyright to narrow Ok(val) to the
    success type and Err(err) to the error type.
    """

    class PatcherError(StrEnum):
        PATH_MISSING = "path_missing"
        OTHER = "other"

    def check(result: Result[str, PatcherError]) -> str:
        match result:
            case Ok(msg):
                return msg
            case Err(PatcherError.PATH_MISSING):
                return "missing"
            case Err(PatcherError.OTHER):
                return "other"
            case Err(other_error):
                return str(other_error)

    assert check(Ok("success")) == "success"
    assert check(Err(PatcherError.PATH_MISSING)) == "missing"


@given(st.one_of(st.integers(), st.none()))
def test_transpose_property(val: int | None) -> None:
    """Verify transpose swapping between Result[T | None] and Result[T] | None."""
    res: Result[int | None, str] = Ok(val)
    transposed = res.transpose()
    if val is None:
        assert transposed is None
    else:
        assert transposed == Ok(val)

    res_err: Result[int | None, str] = Err("fail")
    assert res_err.transpose() == res_err


@given(st.integers(), st.text())
def test_product_property(v1: int, v2: str) -> None:
    """Verify product zipping two results into a tuple."""
    assert Ok(v1).product(Ok(v2)) == Ok((v1, v2))
    assert Ok(v1).product(Err(v2)) == Err(v2)
    assert Err(v2).product(Ok(v1)) == Err(v2)


def test_map2_integration() -> None:
    """Verify map2 zipping and mapping logic."""
    assert map2(Ok(1), Ok(2), operator.add) == Ok(3)
    assert map2(Ok(1), Err("fail"), operator.add) == Err("fail")


def test_any_integration() -> None:
    """Verify any_ok returning first success or collection of errors."""
    results: list[Result[int, str]] = [Err("a"), Ok(1), Ok(2)]
    assert any_ok(results) == Ok(1)
    err_results: list[Result[int, str]] = [Err("a"), Err("b")]
    assert any_ok(err_results) == Err(["a", "b"])


def test_catch_context_manager() -> None:
    """Verify catch used as a context manager trapping localized blocks."""

    def run_catch_success(ctx: CatchContext[int, ValueError]) -> None:
        ctx.set(10)
        assert ctx.result == Ok(10)

    ctx_success: CatchContext[int, ValueError]
    with catch(ValueError) as ctx_success:  # pyright: ignore[reportUnknownVariableType]
        run_catch_success(ctx_success)

    # Use nested with or similar to test exception trap
    try:
        with catch(ValueError):
            _ = int("not a number")
    except ValueError:
        pass

    # Real test for trapping
    def check_trapped(ctx: CatchContext[int, ValueError]) -> None:
        assert ctx.result is not None
        assert is_err(ctx.result)

    ctx_trapped: CatchContext[int, ValueError]
    with catch(ValueError) as ctx_trapped:  # pyright: ignore[reportUnknownVariableType]
        _ = int("not a number")
    check_trapped(ctx_trapped)


def test_do_notation_remapping() -> None:
    """Verify automatic error remapping in do_notation."""

    class InternalError(Exception):
        pass

    class DomainError(Exception):
        def __init__(self, cause: Exception) -> None:
            self.cause = cause

    @do_notation(remap={InternalError: DomainError})
    def workflow() -> Do[Any, Any]:
        # Use explicit type to help basedpyright with union conversion during yield
        err_val: Result[int, DomainError | InternalError] = Err(InternalError("database down"))
        yield err_val
        return 1

    res: Result[int, Exception] = workflow()
    assert is_err(res)
    assert isinstance(res.err(), DomainError)


def test_unwrap_or_default_api() -> None:
    """Verify unwrap_or_default on both variants."""
    val = 10
    assert Ok(val).unsafe.unwrap_or_default() == val
    # Currently unwrap_or_default on Err returns None as a placeholder
    assert Err("fail").unsafe.unwrap_or_default() is None


def test_catch_call_integration_mapping() -> None:
    """Integration: Verify catch_call mapping in a realistic scenario."""

    class ServiceError(StrEnum):
        NOT_FOUND = "not_found"
        DB_ERROR = "db_error"

    data: dict[str, int] = {"a": 1}

    # 1. Simple catch
    assert catch_call(json.JSONDecodeError, json.loads, '{"a": 1}') == Ok({"a": 1})
    assert is_err(catch_call(json.JSONDecodeError, json.loads, "invalid"))

    def get_val(k: str) -> int:
        return data[k]

    # 2. Using map_to for single exception
    res = catch_call(KeyError, get_val, "b", map_to=ServiceError.NOT_FOUND)
    assert res == Err(ServiceError.NOT_FOUND)

    # 3. Using explicit mapping dict
    mapping: dict[type[Exception], Any] = {KeyError: ServiceError.NOT_FOUND, ValueError: ServiceError.DB_ERROR}
    res2 = catch_call(mapping, get_val, "b")
    assert res2 == Err(ServiceError.NOT_FOUND)

    # 4. Successful path preserves Ok
    assert catch_call(KeyError, get_val, "a", map_to=ServiceError.NOT_FOUND) == Ok(1)


def test_as_err_pinpoint() -> None:
    """Verify as_err for manual exception lifting."""
    # 1. Simple lift
    e = ValueError("fail")
    assert as_err(e) == Err(e)

    # 2. Lift with dict mapping
    assert as_err(e, {ValueError: "mapped"}) == Err("mapped")
    assert as_err(e, {TypeError: "wrong"}) == Err(e)

    # 3. Lift with type and map_to
    assert as_err(e, ValueError, map_to="mapped") == Err("mapped")
    assert as_err(e, TypeError, map_to="mapped") == Err(e)

    # 4. Lift with tuple mapping
    assert as_err(e, (ValueError, TypeError), map_to="mapped") == Err("mapped")


def test_partition_exceptions_api() -> None:
    """Verify partition_exceptions correctly separating values from exceptions."""
    items: list[int | Exception] = [1, ValueError("a"), 2, KeyError("b")]
    oks: list[Ok[int]]
    errs: list[Err[Exception]]
    oks, errs = cast("tuple[list[Ok[int]], list[Err[Exception]]]", partition_exceptions(items))  # pyright: ignore[reportUnnecessaryCast]
    assert oks == [Ok(1), Ok(2)]
    expected_err_count = 2
    assert len(errs) == expected_err_count
    assert isinstance(errs[0].err(), ValueError)


# --- Combinators API (Compiler Pipeline Theme) ---


def test_validate_accumulation() -> None:
    """Verify validate accumulates multiple errors or returns tuple of values."""

    def typecheck(val: Any) -> Result[str, str]:
        return Ok("int") if isinstance(val, int) else Err(f"Not an int: {val}")

    # Success case: All Ok -> Ok(tuple)
    res = validate(typecheck(1), typecheck(2))
    assert res == Ok(("int", "int"))

    # Failure case: Multiple Errs -> Err(list)
    res_err = validate(typecheck(1), typecheck("a"), typecheck("b"))
    assert res_err == Err(["Not an int: a", "Not an int: b"])


def test_traverse_fast_fail() -> None:
    """Verify traverse maps over list but fails fast on first error."""

    def parse_stmt(s: str) -> Result[str, str]:
        return Ok(f"AST({s})") if s != "!" else Err("Syntax Error")

    assert traverse(["a", "b"], parse_stmt) == Ok(["AST(a)", "AST(b)"])
    assert traverse(["a", "!", "b"], parse_stmt) == Err("Syntax Error")


def test_try_fold_symbol_table() -> None:
    """Verify try_fold reduction with fallible steps."""

    def add_to_table(table: dict[str, str], pair: tuple[str, str]) -> Result[dict[str, str], str]:
        name, type_ = pair
        if name in table:
            return Err(f"Duplicate: {name}")
        table[name] = type_
        return Ok(table)

    initial: dict[str, str] = {}
    items = [("x", "int"), ("y", "float")]
    assert try_fold(items, initial, add_to_table) == Ok({"x": "int", "y": "float"})

    bad_items = [("x", "int"), ("x", "float")]
    assert is_err(try_fold(bad_items, initial, add_to_table))


def test_ensure_guards() -> None:
    """Verify ensure lifting booleans into Results."""
    expected_sum = 2
    incorrect_sum = 3
    assert ensure(expected_sum == 1 + 1, "logic fail") == Ok(None)
    assert ensure(incorrect_sum == 1 + 1, "math fail") == Err("math fail")


@given(st.one_of(st.text(), st.lists(st.text())), st.text())
def test_add_context_polymorphism_property(err_data: str | list[str], context: str) -> None:
    """Invariant: add_context applies breadcrumbs to strings and collections correctly."""
    res = Err(err_data)
    ctx_res: Result[Any, Any] = add_context(res, context)

    if isinstance(err_data, list):
        assert isinstance(ctx_res.err(), list)
        # Check that every element was transformed
        for original, transformed in zip(err_data, cast("list[str]", ctx_res.err()), strict=True):
            assert transformed == f"{context}: {original}"
    else:
        assert ctx_res.err() == f"{context}: {err_data}"


def test_flow_pipeline() -> None:
    """Verify flow pipes data through sequential fallible steps."""

    def inc(x: int) -> Result[int, str]:
        return Ok(x + 1)

    def to_str(x: int) -> Result[str, str]:
        return Ok(str(x))

    assert flow(10, inc, to_str) == Ok("11")
    assert flow(10, lambda _: Err("fail"), to_str) == Err("fail")


def test_succeeds_filtering() -> None:
    """Verify succeeds extracts only Ok values."""
    results: list[Result[int, str]] = [Ok(1), Err("fail"), Ok(2)]
    assert succeeds(results) == [1, 2]


def test_result_map_exc() -> None:
    """Verify Result.map_exc transforming error payloads."""

    class ErrorCode(StrEnum):
        INVALID = "invalid"
        MISSING = "missing"

    mapping: Mapping[type[Exception], Any] = {ValueError: ErrorCode.INVALID, KeyError: ErrorCode.MISSING}

    # Err matches
    assert Err(ValueError("a")).map_exc(mapping) == Err(ErrorCode.INVALID)
    assert Err(KeyError("b")).map_exc(mapping) == Err(ErrorCode.MISSING)

    # Err no match
    err_other: Result[int, Exception] = Err(RuntimeError("c"))
    assert err_other.map_exc(mapping) == err_other

    # Ok ignored
    assert Ok(10).map_exc(mapping) == Ok(10)


def test_catch_mapping_api() -> None:
    """Verify catch with explicit mapping dictionary or map_to."""

    class ErrorCode(StrEnum):
        INVALID = "invalid"
        MISSING = "missing"

    # 1. With Mapping dict
    @catch({ValueError: ErrorCode.INVALID, KeyError: ErrorCode.MISSING})
    def risky(x: str) -> str:
        if x == "val":
            raise ValueError
        if x == "key":
            raise KeyError
        return "ok"

    res_val = risky("val")
    assert is_err(res_val) and cast("Any", res_val.err()) == ErrorCode.INVALID
    res_key = risky("key")
    assert is_err(res_key) and cast("Any", res_key.err()) == ErrorCode.MISSING
    assert risky("other") == Ok("ok")

    # 2. With map_to
    @catch(ValueError, map_to=ErrorCode.INVALID)
    def simple(x: str) -> int:
        if x == "fail":
            raise ValueError
        return 1

    res_fail = simple("fail")
    assert is_err(res_fail) and cast("Any", res_fail.err()) == ErrorCode.INVALID
    assert simple("ok") == Ok(1)

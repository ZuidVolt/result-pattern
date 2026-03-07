from typing import TYPE_CHECKING, Any, Never, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.result import (
    Do,
    DoAsync,
    Err,
    Ok,
    Result,
    combine,
    do,
    do_async,
    do_notation,
    do_notation_async,
    from_optional,
    is_err,
    is_ok,
    partition,
    safe,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

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
        return res_val  # noqa: B901

    assert chained == procedural(val)


# --- Complex Integration Tests ---


def test_data_pipeline_integration() -> None:
    """Integration: Combine @safe, combine, and @do_notation in a realistic pipeline.

    This test simulates a multi-step data processing flow where external
    exceptions are lifted into Results, combined into a collection, and
    processed imperatively.
    """

    @safe(ValueError)
    def parse_int(s: str) -> int:
        return int(s)

    @do_notation()
    def sum_raw_inputs(raw_inputs: list[str]) -> Do[int, Exception]:
        # 1. Lift exceptions into Result containers
        results = [parse_int(i) for i in raw_inputs]

        # 2. Transpose list of Results into Result of list (all-or-nothing)
        ints: list[int] = yield combine(results)

        # 3. Sum the result
        return sum(ints)  # noqa: B901

    assert sum_raw_inputs(["1", "2", "3"]) == Ok(6)

    res_err: Result[int, Exception] = sum_raw_inputs(["1", "not a number", "3"])
    assert is_err(res_err)


@pytest.mark.asyncio
async def test_async_integration_flow() -> None:
    """Integration: Test complex async flows with do_notation_async and safe wrappers."""

    @safe(RuntimeError)
    async def fetch_user_name(user_id: int) -> str:  # noqa: RUF029
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
        return val  # noqa: B901

    assert risky("10") == Ok(10)
    res: Result[int, Exception] = risky("not a number")
    assert is_err(res)


@pytest.mark.asyncio
async def test_do_notation_async_catch_basic() -> None:
    """Verify @do_notation_async catch parameter lifts exceptions into Err."""

    @do_notation_async(catch=ValueError)
    async def risky_async(s: str) -> DoAsync[int, Exception]:  # noqa: RUF029
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

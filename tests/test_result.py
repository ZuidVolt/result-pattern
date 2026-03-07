from typing import TYPE_CHECKING, Any, Never

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.result import (
    Do,
    DoAsync,
    Err,
    Ok,
    OkErr,
    Result,
    UnwrapError,
    _DoError,  # pyright: ignore[reportPrivateUsage] # noqa: PLC2701
    _raise_api_error,  # pyright: ignore[reportPrivateUsage] # noqa: PLC2701
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
    from collections.abc import AsyncGenerator

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
        res = yield g(b)
        return res  # noqa: B901

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
        ints = yield combine(results)

        # 3. Sum the result
        return sum(ints)  # noqa: B901

    assert sum_raw_inputs(["1", "2", "3"]) == Ok(6)

    res_err = sum_raw_inputs(["1", "not a number", "3"])  # pyright: ignore[reportUnknownVariableType]
    assert is_err(res_err)
    assert isinstance(res_err.unsafe.unwrap_err(), ValueError)  # pyright: ignore[reportUnknownMemberType]


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
        name = yield await fetch_user_name(user_id)
        yield Ok(f"Welcome, {name}!")

    assert await get_welcome_message(1) == Ok("Welcome, User_1!")
    res_err = await get_welcome_message(-1)
    assert is_err(res_err)
    assert str(res_err.unsafe.unwrap_err()) == "invalid id"


# --- Property-Based Coverage ---


@given(st.one_of(st.integers(), st.none()), st.text())
def test_from_optional_invariant(val: int | None, err: str) -> None:
    """Invariant: from_optional produces Ok(val) if not None, else Err(err)."""
    res = from_optional(val, err)
    if val is None:
        assert is_err(res)
        assert res.ok() is None
        assert res.err() == err
    else:
        assert is_ok(res)
        assert res.ok() == val
        assert res.err() is None
        assert res.unsafe.unwrap() == val


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


# --- API Safeguards & Safety ---


@pytest.mark.asyncio
@given(st.integers())
async def test_exhaustive_variant_behavior_property(val: int) -> None:
    """Exhaustively verify variant-specific methods and unsafe proxies."""
    res_ok = Ok(val)
    res_err = Err(val)

    # 1. Functional Async (Direct calls for coverage)
    async def dummy(x: Any) -> Any:  # noqa: RUF029, ANN401
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
            _ = variant.value  # noqa: SLF001

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


# --- Core Do-Notation Utilities ---


def test_do_inline() -> None:
    """Invariant: do() helper correctly unwraps sync generator expressions."""
    assert do(Ok(x + y) for x in Ok(1) for y in Ok(2)) == Ok(3)
    assert do(Ok(x + y) for x in Ok(1) for y in Err("fail")) == Err("fail")


@pytest.mark.asyncio
async def test_do_inline_async() -> None:
    """Invariant: do_async() helper correctly unwraps async generator expressions."""
    res = await do_async(Ok(x + 2) async for x in Ok(1))  # pyright: ignore[reportUnknownVariableType]
    assert res == Ok(3)


def test_do_notation_catch_basic() -> None:
    """Verify @do_notation catch parameter lifts exceptions into Err."""

    @do_notation(catch=ValueError)
    def risky(s: str) -> Do[int, Exception]:
        val = yield Ok(int(s))
        return val  # noqa: B901

    assert risky("10") == Ok(10)
    # Bypass deep union inference issues in basedpyright
    res = risky("not a number")  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(res, Err)

    assert isinstance(res.unsafe.unwrap_err(), ValueError)  # pyright: ignore[reportUnknownMemberType]


@pytest.mark.asyncio
async def test_do_notation_async_catch_basic() -> None:
    """Verify @do_notation_async catch parameter lifts exceptions into Err."""

    @do_notation_async(catch=ValueError)
    async def risky_async(s: str) -> DoAsync[int, Exception]:  # noqa: RUF029
        val = yield Ok(int(s))
        yield Ok(val)

    assert await risky_async("10") == Ok(10)
    res = await risky_async("not a number")
    assert is_err(res)
    assert isinstance(res.unsafe.unwrap_err(), ValueError)


def test_ok_err_constant() -> None:
    """Verify the OkErr convenience constant for isinstance checks."""
    assert isinstance(Ok(1), OkErr)
    assert isinstance(Err(1), OkErr)
    assert not isinstance(1, OkErr)


def test_covariance_support() -> None:
    """Verify true covariance support (Dog -> Animal)."""

    class Animal:
        pass

    class Dog(Animal):
        pass

    # Ok[Dog] should be assignable to Result[Animal, Any]
    res: Result[Animal, Any] = Ok(Dog())
    assert is_ok(res)


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
    async def dummy(x: Any) -> Any:  # noqa: RUF029, ANN401
        return x

    assert await res_err.map_async(dummy) == Err("error")
    assert await res_err.tap_async(dummy) == Err("error")
    assert await res_err.and_then_async(lambda x: dummy(Ok(x))) == Err("error")

    # 3. Empty Generators
    async def empty_gen() -> AsyncGenerator[Result[int, str], Any]:  # noqa: RUF029
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
                m((lambda x, m_local=method: x if "inspect" in m_local else "msg"))  # pyright: ignore[reportUnknownLambdaType]

        with pytest.raises(AttributeError, match=r"API Warning"):
            m = getattr(res_err, method)
            if method in {"unwrap", "unwrap_err"}:
                m()
            else:
                # Bind method variable to avoid B023
                m((lambda x, m_local=method: x if "inspect" in m_local else "msg"))  # pyright: ignore[reportUnknownLambdaType]

    # 6. Err variant trivial methods
    assert res_err.ok() is None
    assert res_err.err() == "error"
    assert res_err.flatten() == res_err
    assert res_err.filter(lambda _: True, "fail") == res_err
    assert res_err.match(on_ok=lambda x: x, on_err=lambda e: e) == "error"

    # 7. Decorator exception paths
    @do_notation(catch=ValueError)
    def fail_sync() -> Do[int, str]:
        msg = "boom"
        raise ValueError(msg)
        yield Ok(1)

    assert is_err(fail_sync())

    @do_notation_async(catch=ValueError)
    async def fail_async() -> DoAsync[int, str]:  # noqa: RUF029
        msg = "boom async"
        raise ValueError(msg)
        yield Ok(1)

    assert is_err(await fail_async())

    @safe(ValueError)
    async def safe_async_fail() -> Never:  # noqa: RUF029
        msg = "safe boom"
        raise ValueError(msg)

    assert is_err(await safe_async_fail())


def test_api_error_default() -> None:
    """Hit the default case in _raise_api_error for unsupported methods."""
    with pytest.raises(AttributeError, match="not part of the supported Result API"):
        _raise_api_error("nonexistent_method")

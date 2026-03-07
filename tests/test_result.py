import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.result import (
    Do,
    DoAsync,
    Err,
    Ok,
    Result,
    UnwrapError,
    do,
    do_async,
    do_notation,
    do_notation_async,
    from_optional,
    is_err,
    is_ok,
    partition,
)


@pytest.mark.asyncio
async def test_do_notation_async_basic() -> None:
    @do_notation_async
    async def calc() -> DoAsync[int, str]:  # noqa: RUF029
        x = yield Ok(1)
        yield Ok(x + 2)

    assert await calc() == Ok(3)


@pytest.mark.asyncio
async def test_do_notation_async_short_circuit() -> None:
    @do_notation_async
    async def calc() -> DoAsync[int, str]:  # noqa: RUF029
        yield Ok(1)
        yield Err("fail")

    assert await calc() == Err("fail")


@pytest.mark.asyncio
async def test_do_notation_async_catch() -> None:
    @do_notation_async(catch=ValueError)
    async def calc() -> DoAsync[int, str]:  # noqa: RUF029
        msg = "boom"
        raise ValueError(msg)
        yield Ok(1)

    res = await calc()
    assert is_err(res)
    assert isinstance(res._error, ValueError)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


def test_do_notation_basic() -> None:
    @do_notation
    def calc() -> Do[int, str]:
        x = yield Ok(1)
        y = yield Ok(2)
        return x + y  # noqa: B901

    assert calc() == Ok(3)


def test_do_notation_short_circuit() -> None:
    @do_notation
    def calc() -> Do[int, str]:
        yield Ok(1)
        yield Err("fail")
        return 0  # noqa: B901

    assert calc() == Err("fail")


def test_do_notation_catch() -> None:
    @do_notation(catch=ValueError)
    def calc() -> Do[int, str]:
        msg = "boom"
        raise ValueError(msg)

    res = calc()
    assert is_err(res)
    assert isinstance(res._error, ValueError)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    assert str(res._error) == "boom"  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


def test_do_notation_no_catch() -> None:
    @do_notation
    def calc() -> Do[int, str]:
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        calc()


def test_do_inline() -> None:
    assert do(Ok(x + y) for x in Ok(1) for y in Ok(2)) == Ok(3)
    assert do(Ok(x + y) for x in Ok(1) for y in Err("fail")) == Err("fail")


@pytest.mark.asyncio
async def test_do_inline_async() -> None:
    # This is how we test the async branch of do_async().
    # Ok(1).__aiter__() will be called, yielding 1 to x.
    res: Result[int, str] = await do_async(Ok(x + 2) async for x in Ok(1))
    assert res == Ok(3)


def test_from_optional() -> None:
    assert from_optional(42, "fail") == Ok(42)
    assert from_optional(None, "fail") == Err("fail")


@given(st.one_of(st.integers(), st.none()), st.text())
def test_from_optional_property(val: int | None, err: str) -> None:
    res = from_optional(val, err)
    if val is None:
        assert is_err(res)
        assert res._error == err  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    else:
        assert is_ok(res)
        assert res._value == val  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


def test_ok_expect() -> None:
    val = "hello"
    res = Ok(val)
    assert res.unsafe.expect("should not fail") == val


def test_err_expect() -> None:
    err = "something went wrong"
    res = Err(err)
    with pytest.raises(UnwrapError, match=r"custom message: 'something went wrong'"):
        res.unsafe.expect("custom message")


@given(st.text())
def test_expect_property(s: str) -> None:
    res = Ok(s)
    assert res.unsafe.expect("msg") == s


@given(st.text(), st.text())
def test_err_expect_property(msg: str, err: str) -> None:
    res = Err(err)
    with pytest.raises(UnwrapError) as exc_info:
        res.unsafe.expect(msg)
    assert repr(err) in str(exc_info.value)
    assert str(msg) in str(exc_info.value)


def test_match() -> None:
    assert Ok(10).match(on_ok=lambda v: v * 2, on_err=lambda _: 0) == 20  # noqa: PLR2004
    assert Err("fail").match(on_ok=lambda v: v, on_err=lambda _: 0) == 0


def test_or_else() -> None:
    assert Ok(1).or_else(lambda _: Ok(2)) == Ok(1)
    assert Err(1).or_else(lambda e: Ok(e + 1)) == Ok(2)
    assert Err(1).or_else(lambda e: Err(e + 1)) == Err(2)


@given(st.integers(), st.integers())
def test_or_else_property(val: int, fallback: int) -> None:
    # Ok stays Ok
    assert Ok(val).or_else(lambda _: Ok(fallback)) == Ok(val)
    # Err recovers or stays Err
    assert Err(val).or_else(lambda _: Ok(fallback)) == Ok(fallback)
    assert Err(val).or_else(lambda _: Err(fallback)) == Err(fallback)


# --- New Utility Tests ---


@given(st.integers(), st.text())
def test_map_or(val: int, default: str) -> None:
    assert Ok(val).map_or(default, str) == str(val)
    assert Err(val).map_or(default, str) == default


@given(st.integers(), st.text())
def test_map_or_else(val: int, default: str) -> None:
    assert Ok(val).map_or_else(lambda: default, str) == str(val)
    assert Err(val).map_or_else(lambda: default, str) == default


def test_unwrap_error_context() -> None:
    res = Err("context")
    with pytest.raises(UnwrapError) as exc_info:
        res.unsafe.unwrap()
    assert exc_info.value.result == res


@given(st.integers(), st.text())
def test_replace(val: int, new_val: str) -> None:
    assert Ok(val).replace(new_val) == Ok(new_val)
    assert Err(val).replace(new_val) == Err(val)


@given(st.integers(), st.text())
def test_replace_err(val: int, new_err: str) -> None:
    assert Ok(val).replace_err(new_err) == Ok(val)
    assert Err(val).replace_err(new_err) == Err(new_err)


def test_tap_err() -> None:
    side_effect: list[str] = []
    Ok(10).tap_err(lambda e: side_effect.append(str(e)))
    assert not side_effect

    Err("fail").tap_err(lambda e: side_effect.append(str(e)))
    assert side_effect == ["fail"]


def test_tap_aliases() -> None:
    side_effect: list[int] = []
    val = 10
    res_ok = Ok(val)
    res_ok.tap(side_effect.append)
    assert side_effect == [val]

    res_err = Err("fail")
    res_err.tap_err(lambda _e: side_effect.append(99))
    assert side_effect == [val, 99]

    # Verify legacy inspect methods now raise AttributeErrors
    with pytest.raises(AttributeError, match=r"inspect"):
        res_ok.inspect(side_effect.append)  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(AttributeError, match=r"inspect_err"):
        res_err.inspect_err(lambda _e: side_effect.append(100))  # pyright: ignore[reportAttributeAccessIssue]


@given(st.integers())
def test_tap_property(val: int) -> None:
    side_effect: list[int] = []
    res = Ok(val).tap(side_effect.append)
    assert side_effect == [val]
    assert res == Ok(val)


@given(st.text())
def test_tap_err_property(err: str) -> None:
    side_effect: list[str] = []
    res = Err(err).tap_err(side_effect.append)
    assert side_effect == [err]
    assert res == Err(err)


def test_unwrap_or_raise() -> None:
    val = 10
    assert Ok(val).unsafe.unwrap_or_raise(ValueError) == val
    with pytest.raises(ValueError, match="boom"):
        Err("boom").unsafe.unwrap_or_raise(ValueError)


@given(st.integers())
def test_unwrap_or_raise_property(val: int) -> None:
    assert Ok(val).unsafe.unwrap_or_raise(ValueError) == val


@given(st.text())
def test_unwrap_or_raise_err_property(err: str) -> None:
    class CustomError(Exception):
        pass

    with pytest.raises(CustomError) as exc_info:
        Err(err).unsafe.unwrap_or_raise(CustomError)
    assert str(exc_info.value) == err


def test_flatten() -> None:
    assert Ok(Ok(10)).flatten() == Ok(10)
    assert Ok(Err("fail")).flatten() == Err("fail")
    assert Ok(10).flatten() == Ok(10)
    assert Err("fail").flatten() == Err("fail")


@given(st.integers(), st.text())
def test_filter(val: int, err: str) -> None:
    assert Ok(val).filter(lambda x: x > 0, err) == (Ok(val) if val > 0 else Err(err))
    assert Err(val).filter(lambda x: x > 0, err) == Err(val)


def test_unsafe_expect() -> None:
    val = 10
    assert Ok(val).unsafe.expect("fail") == val
    with pytest.raises(UnwrapError, match=r"fail: 'boom'"):
        Err("boom").unsafe.expect("fail")


@given(st.lists(st.booleans()))
def test_partition_property(bools: list[bool]) -> None:
    results = [Ok(i) if b else Err(f"err_{i}") for i, b in enumerate(bools)]
    oks, errs = partition(results)
    assert len(oks) == sum(bools)
    assert len(errs) == len(bools) - sum(bools)
    assert len(oks) + len(errs) == len(results)


def test_variant_safeguards() -> None:
    val = 10
    res_ok: Result[int, str] = Ok(val)
    res_err: Result[int, str] = Err("fail")

    with pytest.raises(AttributeError, match=r"Direct access to '.value' is disabled"):
        _ = res_ok.value  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(AttributeError, match=r"Direct access to '.error' is disabled"):
        _ = res_err.error  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(AttributeError, match=r"is a crashing operation and is isolated in the '.unsafe' namespace"):
        _ = res_ok.unwrap_or_raise(ValueError)  # pyright: ignore[reportAttributeAccessIssue]


def test_ok_err_conversion() -> None:
    val = 10
    res_ok = Ok(val)
    res_err = Err("fail")

    assert res_ok.ok() == val
    assert res_ok.err() is None

    assert res_err.ok() is None
    assert res_err.err() == "fail"


def test_namespaced_unsafe() -> None:
    val = 10
    res_ok = Ok(val)
    res_err = Err("fail")

    assert res_ok.unsafe.unwrap() == val
    assert res_err.unsafe.unwrap_err() == "fail"

    with pytest.raises(UnwrapError):
        res_ok.unsafe.unwrap_err()

    with pytest.raises(UnwrapError):
        res_err.unsafe.unwrap()

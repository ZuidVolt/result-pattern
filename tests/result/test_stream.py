import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from result import Err, Ok, SafeStream, SafeStreamAsync, catch_each_iter, catch_each_iter_async, is_err

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

# --- Sync Tests ---


def test_catch_iter_sync_success() -> None:
    @catch_each_iter(ValueError)
    def count(n: int) -> Generator[int, Any, Any]:
        yield from range(n)

    stream: SafeStream[int, ValueError] = count(3)
    items = list(stream)
    assert items == [Ok(0), Ok(1), Ok(2)]


def test_catch_iter_sync_failure() -> None:
    limit = 2
    expected_len = 3

    @catch_each_iter(ValueError)
    def count_fail(n: int) -> Generator[int, Any, Any]:
        for i in range(n):
            if i == limit:
                raise ValueError("too big")
            yield i

    stream: SafeStream[int, ValueError] = count_fail(5)
    items = list(stream)
    assert len(items) == expected_len
    assert items[0] == Ok(0)
    assert items[1] == Ok(1)
    assert is_err(items[2])
    assert str(items[2].err()) == "too big"


def test_safe_stream_to_result() -> None:
    @catch_each_iter(ValueError)
    def gen(*, fail: bool) -> Generator[int, Any, Any]:
        yield 1
        if fail:
            raise ValueError("fail")
        yield 2

    # 1. Success case
    stream_ok: SafeStream[int, ValueError] = gen(fail=False)
    res_ok = stream_ok.to_result()
    assert res_ok == Ok([1, 2])

    # 2. Failure case
    stream_err: SafeStream[int, ValueError] = gen(fail=True)
    res_err = stream_err.to_result()
    assert is_err(res_err)


def test_safe_stream_to_outcome() -> None:
    @catch_each_iter(ValueError)
    def gen() -> Generator[int, Any, Any]:
        yield 1
        raise ValueError("fail")
        # No yield here to avoid unreachable code warning

    stream: SafeStream[int, ValueError] = gen()
    out = stream.to_outcome()
    assert out.value == [1]
    assert isinstance(out.error, list)
    assert len(out.error) == 1  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert isinstance(out.error[0], ValueError)  # pyright: ignore[reportUnknownMemberType]


def test_safe_stream_one_shot() -> None:
    @catch_each_iter(ValueError)
    def gen() -> Generator[int, Any, Any]:
        yield 1

    stream: SafeStream[int, ValueError] = gen()
    list(stream)
    with pytest.raises(RuntimeError, match=r"SafeStream can only be iterated once"):
        list(stream)


# --- Async Tests ---


@pytest.mark.asyncio
async def test_catch_iter_async_success() -> None:
    @catch_each_iter_async(ValueError)
    async def count_async(n: int) -> AsyncGenerator[int, Any]:
        for i in range(n):
            await asyncio.sleep(0)
            yield i

    stream: SafeStreamAsync[int, ValueError] = count_async(3)
    items = [res async for res in stream]
    assert items == [Ok(0), Ok(1), Ok(2)]


@pytest.mark.asyncio
async def test_catch_iter_async_failure() -> None:
    limit = 1
    expected_len = 2

    @catch_each_iter_async(ValueError)
    async def count_fail_async(n: int) -> AsyncGenerator[int, Any]:
        for i in range(n):
            if i == limit:
                await asyncio.sleep(0)
                raise ValueError("async fail")
            yield i

    stream: SafeStreamAsync[int, ValueError] = count_fail_async(3)
    items = [res async for res in stream]

    assert len(items) == expected_len
    assert items[0] == Ok(0)
    assert is_err(items[1])


@pytest.mark.asyncio
async def test_safe_stream_async_conversions() -> None:
    @catch_each_iter_async(ValueError)
    async def gen_async(*, fail: bool) -> AsyncGenerator[int, Any]:
        yield 1
        if fail:
            await asyncio.sleep(0)
            raise ValueError("fail")
        yield 2

    # 1. to_result
    stream_ok: SafeStreamAsync[int, ValueError] = gen_async(fail=False)
    assert await stream_ok.to_result() == Ok([1, 2])

    stream_err: SafeStreamAsync[int, ValueError] = gen_async(fail=True)
    res_err = await stream_err.to_result()
    assert is_err(res_err)

    # 2. to_outcome
    stream_out: SafeStreamAsync[int, ValueError] = gen_async(fail=True)
    out = await stream_out.to_outcome()
    assert out.value == [1]
    assert out.error is not None
    assert len(out.error) == 1  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


# --- Mapping Tests ---


def test_catch_iter_mapping() -> None:
    @catch_each_iter(ValueError, map_to="mapped_err")
    def gen() -> Generator[int, Any, Any]:
        yield 1
        raise ValueError("original")

    stream: SafeStream[int, Any] = gen()
    items = list(stream)
    assert items[1] == Err("mapped_err")

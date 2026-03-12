import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from result import Err, Ok, Result, catch, retry_result, retry_result_async


def test_retry_sync_success() -> None:
    calls = 0
    attempts = 3

    @retry_result(attempts=attempts)
    def op() -> Ok[int] | Err[str]:
        nonlocal calls
        calls += 1
        expected_calls_for_success = 2
        if calls < expected_calls_for_success:
            return Err("fail")
        return Ok(42)

    res: Result[Any, Any] = op()
    assert res == Ok(42)
    expected_total_calls = 2
    assert calls == expected_total_calls


def test_retry_sync_exhausted() -> None:
    calls = 0
    attempts = 3

    @retry_result(attempts=attempts)
    def op() -> Err[str]:
        nonlocal calls
        calls += 1
        return Err("fail")

    res: Result[Any, str] = op()  # pyright: ignore[reportUnknownVariableType]
    assert res == Err("fail")
    assert calls == attempts


def test_retry_if_predicate() -> None:
    calls = 0
    attempts_limit = 5

    @retry_result(attempts=attempts_limit, retry_if=lambda e: e == "retryable")
    def op() -> Err[str]:
        nonlocal calls
        calls += 1
        first_call = 1
        if calls == first_call:
            return Err("retryable")
        return Err("fatal")

    res: Result[Any, str] = op()  # pyright: ignore[reportUnknownVariableType]
    assert res == Err("fatal")
    expected_calls = 2
    assert calls == expected_calls


def test_retry_with_catch_sync() -> None:
    calls = 0
    attempts = 3

    @retry_result(attempts=attempts, catch=ValueError)
    def op() -> int:
        nonlocal calls
        calls += 1
        expected_calls_for_success = 3
        if calls < expected_calls_for_success:
            raise ValueError("fail")
        return 42

    res: Result[int, Any] = op()  # pyright: ignore[reportUnknownVariableType]
    assert res == Ok(42)
    assert calls == attempts


def test_retry_with_catch_exhausted() -> None:
    calls = 0
    attempts = 3

    @retry_result(attempts=attempts, catch=ValueError)
    def op() -> int:
        nonlocal calls
        calls += 1
        raise ValueError("fail")

    res: Result[int, Any] = op()  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(res, Err)
    err = res.err()
    assert isinstance(err, ValueError)
    assert calls == attempts


def test_retry_plus_catch_stacking() -> None:
    # Verify that putting retry on TOP of catch works without the 'catch' parameter
    calls = 0
    attempts = 3

    @retry_result(attempts=attempts)
    @catch(ValueError)
    def op() -> int:
        nonlocal calls
        calls += 1
        expected_calls_for_success = 2
        if calls < expected_calls_for_success:
            raise ValueError("fail")
        return 42

    res: Result[Any, Any] = op()
    assert res == Ok(42)
    expected_total_calls = 2
    assert calls == expected_total_calls


@pytest.mark.asyncio
async def test_retry_async_success() -> None:
    calls = 0
    attempts = 3

    @retry_result_async(attempts=attempts, delay=0.01)
    async def op() -> Ok[int] | Err[str]:
        nonlocal calls
        calls += 1
        # Use sleep to satisfy async
        await asyncio.sleep(0)
        expected_calls_for_success = 3
        if calls < expected_calls_for_success:
            return Err("transient")
        return Ok(100)

    res: Result[Any, Any] = await op()  # pyright: ignore[reportUnknownVariableType]
    assert res == Ok(100)
    assert calls == attempts


@pytest.mark.asyncio
async def test_retry_with_catch_async() -> None:
    calls = 0
    attempts = 3

    @retry_result_async(attempts=attempts, catch=ValueError)
    async def op() -> int:
        nonlocal calls
        calls += 1
        # use await to avoid RUF029
        await asyncio.sleep(0)
        expected_calls_for_success = 2
        if calls < expected_calls_for_success:
            raise ValueError("async fail")
        return 100

    res: Result[int, Any] = await op()  # pyright: ignore[reportUnknownVariableType]
    assert res == Ok(100)
    expected_total_calls = 2
    assert calls == expected_total_calls


def test_retry_backoff_logic() -> None:
    with patch("time.sleep") as mock_sleep:
        attempts = 3
        initial_delay = 0.1
        backoff_factor = 2.0

        @retry_result(attempts=attempts, delay=initial_delay, backoff=backoff_factor)
        def op() -> Err[str]:
            return Err("fail")

        op()
        expected_sleep_count = 2
        assert mock_sleep.call_count == expected_sleep_count
        mock_sleep.assert_any_call(0.1)
        mock_sleep.assert_any_call(0.2)

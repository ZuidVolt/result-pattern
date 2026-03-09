import asyncio
from typing import Any

import pytest

from result import (
    Err,
    Ok,
    Outcome,
    Result,
    gather_outcomes,
    gather_results,
    partition_results,
    traverse_async,
    traverse_async_outcome,
    validate_async,
)

# --- Mock Workers ---


async def delayed_ok[T](val: T, delay: float = 0.01) -> Result[T, Any]:
    await asyncio.sleep(delay)
    return Ok(val)


async def delayed_err[E](err: E, delay: float = 0.01) -> Result[Any, E]:
    await asyncio.sleep(delay)
    return Err(err)


async def delayed_outcome[T, E](val: T, err: E | None, delay: float = 0.01) -> Outcome[T, E]:  # noqa: RUF029
    await asyncio.sleep(delay)
    return Outcome(val, err)


# --- Result Concurrency Tests ---


@pytest.mark.asyncio
async def test_gather_results_success() -> None:
    """Verify gather_results aggregates values in order."""
    res = await gather_results(delayed_ok(1, 0.02), delayed_ok(2, 0.01))
    assert res == Ok([1, 2])


@pytest.mark.asyncio
async def test_gather_results_fail_fast() -> None:
    """Verify gather_results returns the first Err encounterd."""
    # The error is faster than the success
    res = await gather_results(delayed_ok(1, 0.05), delayed_err("fail", 0.01))
    assert res == Err("fail")


@pytest.mark.asyncio
async def test_validate_async_accumulation() -> None:
    """Verify validate_async waits for everyone and accumulates all errors."""
    res = await validate_async(delayed_err("e1", 0.01), delayed_ok(1, 0.02), delayed_err("e2", 0.03))
    # It must wait for the 0.03s task and return BOTH errors
    assert res == Err(["e1", "e2"])


@pytest.mark.asyncio
async def test_traverse_async_monadic() -> None:
    """Verify traverse_async behaves like a concurrent monadic map."""

    async def process(n: int) -> Result[int, str]:  # noqa: RUF029
        return Ok(n * 2) if n > 0 else Err(f"neg: {n}")

    # Success
    res = await traverse_async([1, 2, 3], process)
    assert res == Ok([2, 4, 6])

    # Failure
    res_err = await traverse_async([1, -1, 2], process)
    assert res_err == Err("neg: -1")


@pytest.mark.asyncio
async def test_traverse_async_limit() -> None:
    """Verify traverse_async respects the concurrency limit."""
    running = 0
    max_observed = 0

    async def worker(_: int) -> Ok[None]:
        nonlocal running, max_observed
        running += 1
        max_observed = max(max_observed, running)
        await asyncio.sleep(0.01)
        running -= 1
        return Ok(None)

    # Process 10 items but limit to 2 at a time
    await traverse_async(range(10), worker, limit=2)
    assert max_observed <= 2  # noqa: PLR2004


# --- Outcome Concurrency Tests ---


@pytest.mark.asyncio
async def test_gather_outcomes_merging() -> None:
    """Verify gather_outcomes combines values and flattens diagnostics."""
    res = await gather_outcomes(
        delayed_outcome(1, "w1", 0.01), delayed_outcome(2, None, 0.01), delayed_outcome(3, ["w2", "w3"], 0.01)
    )
    assert res.value == [1, 2, 3]
    assert res.error == ["w1", "w2", "w3"]


@pytest.mark.asyncio
async def test_traverse_async_outcome_flow() -> None:
    """Verify concurrent map-reduce for outcomes."""

    async def lenient_worker(n: int) -> Outcome[int, str]:  # noqa: RUF029
        if n % 2 == 0:
            return Outcome(n, None)
        return Outcome(n, f"odd: {n}")

    res = await traverse_async_outcome([1, 2, 3, 4], lenient_worker)
    assert res.value == [1, 2, 3, 4]
    assert res.error == ["odd: 1", "odd: 3"]


# --- Bridge Tests ---


def test_partition_results_logic() -> None:
    """Verify slicing of mixed result batches."""
    results: list[Result[int, str]] = [Ok(1), Err("e1"), Ok(2), Err("e2")]
    vals: list[int]
    errs: list[str]
    vals, errs = partition_results(results)
    assert vals == [1, 2]
    assert errs == ["e1", "e2"]

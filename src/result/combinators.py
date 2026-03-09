"""# Result Combinators: High-level functional utilities for Result patterns.

This module provides advanced orchestration tools like error accumulation,
short-circuiting iteration, concurrent mapping, and functional pipelining.
"""

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
# pyright: reportOverlappingOverload=false

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast, overload

from .result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from .outcome import Outcome


@overload
def validate[T1, T2, E](r1: Result[T1, E], r2: Result[T2, E]) -> Result[tuple[T1, T2], list[E]]: ...


@overload
def validate[T1, T2, T3, E](
    r1: Result[T1, E], r2: Result[T2, E], r3: Result[T3, E]
) -> Result[tuple[T1, T2, T3], list[E]]: ...


@overload
def validate[T1, T2, T3, T4, E](
    r1: Result[T1, E], r2: Result[T2, E], r3: Result[T3, E], r4: Result[T4, E]
) -> Result[tuple[T1, T2, T3, T4], list[E]]: ...


def validate(*results: Result[Any, Any]) -> Result[tuple[Any, ...], list[Any]]:  # type: ignore[misc]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]
    """Accumulate all errors from multiple results, or return a tuple of all values.

    Unlike standard monadic chaining (.and_then), this does not short-circuit.
    It is the 'Applicative' alternative to fail-fast logic.

    Args:
        *results: Multiple Result instances to validate.

    Returns:
        Ok(tuple) if all are Ok, otherwise Err(list) containing all failures.

    Examples:
        >>> validate(Ok(1), Ok("a"))
        Ok((1, 'a'))

        >>> # Accumulates multiple errors
        >>> validate(Ok(1), Err("fail1"), Err("fail2"))
        Err(['fail1', 'fail2'])

    """
    errors: list[Any] = []
    values: list[Any] = []

    for res in results:
        if isinstance(res, Err):
            errors.append(res._error)
        else:
            values.append(res._value)

    if errors:
        return Err(errors)
    return Ok(tuple(values))


def traverse[T, U, E](items: Iterable[T], func: Callable[[T], Result[U, E]]) -> Result[list[U], E]:
    """Map a fallible function over an iterable, short-circuiting on the first error.

    Ideal for bulk processing where any single failure invalidates the batch.

    Args:
        items: An iterable of items to process.
        func: A function returning a Result for each item.

    Returns:
        Ok(list) of all transformed values, or the first Err encountered.

    Examples:
        >>> traverse(["1", "2"], lambda x: Ok(int(x)))
        Ok([1, 2])

        >>> # Short-circuits on '!'
        >>> traverse(["1", "!", "2"], lambda x: Ok(int(x)) if x.isdigit() else Err(f"bad: {x}"))
        Err('bad: !')

    """
    values: list[U] = []
    for item in items:
        res = func(item)
        if isinstance(res, Err):
            return res
        values.append(res._value)
    return Ok(values)


def try_fold[T, U, E](items: Iterable[T], initial: U, func: Callable[[U, T], Result[U, E]]) -> Result[U, E]:
    """Fallible reduction (fold). Short-circuits on the first error.

    Useful for building state (like a symbol table) from a list of operations.

    Args:
        items: An iterable of items to reduce.
        initial: The starting accumulator value.
        func: A function taking (accumulator, item) and returning a Result.

    Returns:
        The final accumulator value wrapped in Ok, or the first Err encountered.

    Examples:
        >>> try_fold([1, 2, 3], 0, lambda acc, x: Ok(acc + x))
        Ok(6)

        >>> def build_path(acc, part):
        ...     return Ok(f"{acc}/{part}") if part else Err("empty part")
        >>> try_fold(["usr", "", "bin"], "", build_path)
        Err('empty part')

    """
    acc = initial
    for item in items:
        res = func(acc, item)
        if isinstance(res, Err):
            return res
        acc = res._value
    return Ok(acc)


def ensure[E](condition: bool, error: E) -> Result[None, E]:  # noqa: FBT001
    """Lift a boolean condition into a Result.

    Often used inside @do_notation to guard logic.

    Args:
        condition: The boolean check to perform.
        error: The error value to return if the condition is False.

    Returns:
        Ok(None) if True, Err(error) if False.

    Examples:
        >>> ensure(10 > 5, "logic fail")
        Ok(None)
        >>> ensure(1 > 5, "logic fail")
        Err('logic fail')

    """
    return Ok(None) if condition else Err(error)


def add_context[T, E](res: Result[T, E], context: str) -> Result[T, str]:
    """Enrich an error payload with additional context if the result is an Err.

    Args:
        res: The original Result.
        context: The string to prepend to the error.

    Returns:
        The original result if Ok, or a new Err with the context prepended.

    Examples:
        >>> add_context(Err("refused"), "Connection")
        Err('Connection: refused')

    """
    if isinstance(res, Err):
        return Err(f"{context}: {res._error}")
    return res


@overload
def flow[T, T1, E](val: T, f1: Callable[[T], Result[T1, E]]) -> Result[T1, E]: ...


@overload
def flow[T, T1, T2, E](
    val: T,
    f1: Callable[[T], Result[T1, E]],
    f2: Callable[[T1], Result[T2, E]],
) -> Result[T2, E]: ...


@overload
def flow[T, T1, T2, T3, E](
    val: T,
    f1: Callable[[T], Result[T1, E]],
    f2: Callable[[T1], Result[T2, E]],
    f3: Callable[[T2], Result[T3, E]],
) -> Result[T3, E]: ...


def flow(initial: Any, *funcs: Callable[[Any], Result[Any, Any]]) -> Result[Any, Any]:  # type: ignore[misc]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]
    """Sequential pipeline orchestrator. Pipes data through fallible functions.

    Equivalent to chaining .and_then() repeatedly.

    Args:
        initial: The starting value.
        *funcs: Multiple functions that take a value and return a Result.

    Returns:
        The final Result of the pipeline, or the first Err encountered.

    Examples:
        >>> flow("10", lambda x: Ok(int(x)), lambda x: Ok(x * 2))
        Ok(20)

    """
    res: Result[Any, Any] = Ok(initial)
    for func in funcs:
        if isinstance(res, Err):
            return res
        res = func(res._value)
    return res


def succeeds[T, E](results: Iterable[Result[T, E]]) -> list[T]:
    """Filter out all Errs and return a list of all success values.

    Args:
        results: An iterable of Results.

    Returns:
        A list of all values contained in Ok variants.

    Examples:
        >>> succeeds([Ok(1), Err("fail"), Ok(2)])
        [1, 2]

    """
    return [res._value for res in results if isinstance(res, Ok)]


def partition_exceptions[T, E: BaseException](
    items: Iterable[T | E],
) -> tuple[list[Ok[T]], list[Err[E]]]:
    """Partition a mixed iterable of values and Python Exceptions into Oks and Errs.

    Ideal for handling raw results from `asyncio.gather(..., return_exceptions=True)`.

    Args:
        items: An iterable containing either values of type T or Exceptions of type E.

    Returns:
        A tuple of (ok_variants, err_variants).

    Examples:
        >>> items = [1, ValueError("fail"), 2]
        >>> oks, errs = partition_exceptions(items)
        >>> oks[0]
        Ok(1)
        >>> errs[0]
        Err(ValueError('fail'))

    """
    oks: list[Ok[T]] = []
    errs: list[Err[E]] = []

    for item in items:
        if isinstance(item, BaseException):
            errs.append(Err(cast("E", item)))
        else:
            oks.append(Ok(cast("T", item)))  # type: ignore[redundant-cast]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]

    return oks, errs


# --- Asynchronous Combinators ---


async def gather_results[T, E](*coroutines: Awaitable[Result[T, E]]) -> Result[list[T], E]:
    """Monadic 'All-or-Nothing' async concurrency.

    Runs multiple Result-returning tasks concurrently. Resolves to the
    *first* Err encountered, or Ok(list) if all succeed.

    Args:
        *coroutines: Async tasks that return a Result.

    Returns:
        The first Err found, or Ok containing a list of all results in order.

    Examples:
        >>> async def worker(n):
        ...     return Ok(n * 2)
        >>> await gather_results(worker(1), worker(2))
        Ok([2, 4])

    """
    results = await asyncio.gather(*coroutines)
    values: list[T] = []
    for res in results:
        if isinstance(res, Err):
            return res
        values.append(res._value)
    return Ok(values)


async def validate_async[T, E](*coroutines: Awaitable[Result[T, E]]) -> Result[list[T], list[E]]:
    """Applicative async concurrency.

    Waits for ALL tasks to finish. If any failed, returns Err containing
    a list of ALL accumulated errors.

    Args:
        *coroutines: Async tasks that return a Result.

    Returns:
        Ok(list) if all succeed, or Err(list) containing all errors.

    Examples:
        >>> async def fail(msg):
        ...     return Err(msg)
        >>> await validate_async(fail("e1"), fail("e2"))
        Err(['e1', 'e2'])

    """
    results = await asyncio.gather(*coroutines)
    errors: list[E] = []
    values: list[T] = []

    for res in results:
        if isinstance(res, Err):
            errors.append(res._error)
        else:
            values.append(res._value)

    if errors:
        return Err(errors)
    return Ok(values)


async def traverse_async[T, U, E](
    items: Iterable[T],
    func: Callable[[T], Awaitable[Result[U, E]]],
    *,
    limit: int | None = None,
) -> Result[list[U], E]:
    """Concurrently map a fallible async function over an iterable.

    Short-circuits on the first error. Includes an optional semaphore
    limit to prevent resource flooding.

    Args:
        items: An iterable of inputs.
        func: An async function returning a Result.
        limit: Optional concurrency limit (semaphore).

    Returns:
        Ok(list) of transformed values, or the first Err found.

    Examples:
        >>> async def fetch(url):
        ...     return Ok(f"data from {url}")
        >>> await traverse_async(["a.com", "b.com"], fetch, limit=2)
        Ok(['data from a.com', 'data from b.com'])

    """
    if limit is not None:
        sem = asyncio.Semaphore(limit)

        async def worker(item: T) -> Result[U, E]:
            async with sem:
                return await func(item)

        tasks = [worker(item) for item in items]
    else:
        tasks = [func(item) for item in items]  # type: ignore[misc]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]

    return await gather_results(*tasks)


def combine_outcomes[T, E](outcomes: Iterable[Outcome[T, E]]) -> Outcome[list[T], list[E]]:
    """Aggregate a collection of outcomes into a single master Outcome.

    Args:
        outcomes: An iterable of Outcome instances.

    Returns:
        A master Outcome containing all values and a flat list of all errors.

    Examples:
        >>> from result import Outcome
        >>> combine_outcomes([Outcome(1, "err1"), Outcome(2, None)])
        Outcome(value=[1, 2], error=['err1'])

    """
    from .outcome import Outcome  # noqa: PLC0415

    all_values: list[T] = []
    all_errors: list[Any] = []

    for out in outcomes:
        all_values.append(out.value)
        if out.error is not None:
            if isinstance(out.error, list):
                all_errors.extend(out.error)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            else:
                all_errors.append(out.error)

    final_err = all_errors or None
    return Outcome(all_values, final_err)


async def gather_outcomes[T, E](*coroutines: Awaitable[Outcome[T, Any]]) -> Outcome[list[T], list[Any]]:
    """Concurrently await multiple Outcome-returning tasks and merge them.

    Ensures that every task completes and all diagnostics are gathered.

    Args:
        *coroutines: Async tasks returning Outcomes.

    Returns:
        A master Outcome containing all values and combined errors.

    """
    results = await asyncio.gather(*coroutines)
    return combine_outcomes(results)


async def traverse_async_outcome[T, U, E](
    items: Iterable[T],
    func: Callable[[T], Awaitable[Outcome[U, Any]]],
) -> Outcome[list[U], list[Any]]:
    """Concurrent map-reduce for fault-tolerant workloads.

    Args:
        items: An iterable of inputs.
        func: An async function returning an Outcome.

    Returns:
        A master Outcome of the batch.

    """
    tasks = [func(item) for item in items]
    return await gather_outcomes(*tasks)


def partition_results[T, E](results: Iterable[Result[T, E]]) -> tuple[list[T], list[E]]:
    """Procedurally slice a completed batch of Results into values and errors.

    Args:
        results: An iterable of Results.

    Returns:
        A tuple of (values, errors).

    Examples:
        >>> partition_results([Ok(1), Err("fail")])
        ([1], ['fail'])

    """
    values: list[T] = []
    errors: list[E] = []

    for res in results:
        if isinstance(res, Ok):
            values.append(res._value)
        else:
            errors.append(res._error)

    return values, errors

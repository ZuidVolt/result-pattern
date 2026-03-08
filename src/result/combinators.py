"""# Result Combinators: High-level functional utilities for Result patterns.

This module provides advanced orchestration tools like error accumulation,
short-circuiting iteration, and functional pipelining.
"""

# ruff: noqa: SLF001
# pyright: reportPrivateUsage=false
# pyright: reportOverlappingOverload=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from .result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
E2 = TypeVar("E2")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")


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

    Unlike standard chaining, this does not short-circuit on the first error.

    Args:
        *results: Multiple Result instances to validate.

    Returns:
        Ok(tuple) if all are Ok, otherwise Err(list) containing all failures.

    Examples:
        >>> validate(Ok(1), Ok("a"))
        Ok((1, 'a'))

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

    Args:
        items: An iterable of items to process.
        func: A function returning a Result for each item.

    Returns:
        Ok(list) of all transformed values, or the first Err encountered.

    Examples:
        >>> traverse(["1", "2"], lambda x: Ok(int(x)))
        Ok([1, 2])

        >>> traverse(["1", "!"], lambda x: Ok(int(x)) if x.isdigit() else Err("invalid"))
        Err('invalid')

        >>> traverse(["1", "!"], lambda x: catch_call(ValueError, int, x))
        Err(ValueError(...))

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

    Args:
        condition: The boolean check to perform.
        error: The error value to return if the condition is False.

    Returns:
        Ok(None) if True, Err(error) if False.

    Examples:
        >>> ensure(1 + 1 == 2, "logic fail")
        Ok(None)

        >>> ensure(1 + 1 == 3, "math fail")
        Err('math fail')

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
        >>> add_context(Err("not found"), "File IO")
        Err('File IO: not found')

        >>> add_context(Ok(200), "Network")
        Ok(200)

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

    Args:
        initial: The starting value.
        *funcs: Multiple functions that take a value and return a Result.

    Returns:
        The final Result of the pipeline, or the first Err encountered.

    Examples:
        >>> flow(10, lambda x: Ok(x * 2), lambda x: Ok(str(x)))
        Ok('20')

        >>> flow("10", lambda x: Ok(int(x)), lambda _: Err("stop"))
        Err('stop')

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

    This is particularly useful for handling results from batch operations
    like `asyncio.gather(..., return_exceptions=True)`.

    Args:
        items: An iterable containing either values of type T or Exceptions of type E.

    Returns:
        A tuple of (ok_variants, err_variants).

    Examples:
        >>> items = [1, ValueError("fail"), 2]
        >>> oks, errs = partition_exceptions(items)
        >>> oks
        [Ok(1), Ok(2)]
        >>> errs
        [Err(ValueError('fail'))]

    """
    oks: list[Ok[T]] = []
    errs: list[Err[E]] = []

    for item in items:
        if isinstance(item, BaseException):
            errs.append(Err(cast("E", item)))
        else:
            oks.append(Ok(cast("T", item)))  # type: ignore[redundant-cast]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]

    return oks, errs

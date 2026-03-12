"""# Adapters: Bulk lifting and integration utilities for the Result Pattern.

This module provides tools to bridge the gap between third-party imperative
APIs and the functional world of Results. This includes class-level
decorators, instance proxies, and fault-tolerant iteration.

Note:
    Most utilities in this module use dynamic proxying or bulk decoration,
    which can lead to **Type Erasure** in some static analysis tools.

"""

# pyright: reportPrivateUsage=false
# mypy: disable-error-code="no-any-return, redundant-cast"

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from .result import Err, Ok, Result, _resolve_mapping, catch, combine, partition

if TYPE_CHECKING:
    from .outcome import Outcome

T_cls = TypeVar("T_cls", bound=type[Any])
T_obj = TypeVar("T_obj")


def _wrap_gen_sync[T, E: Exception](
    original_gen: Iterator[T],
    catch_tuple: tuple[type[E], ...],
    exc_map: Mapping[type[Exception], Any],
    *,
    has_mapping: bool,
) -> Iterator[Result[T, Any]]:
    """Internal helper to wrap a synchronous iterator with exception handling."""
    try:
        for val in original_gen:
            yield Ok(val)
    except catch_tuple as e:
        yield Err(exc_map.get(type(e), e) if has_mapping else e)
    except Exception as e:  # noqa: BLE001
        tb = e.__traceback__
        raise e.with_traceback(tb.tb_next if tb else None) from None


async def _wrap_gen_async[T, E: Exception](
    original_gen: AsyncIterator[T],
    catch_tuple: tuple[type[E], ...],
    exc_map: Mapping[type[Exception], Any],
    *,
    has_mapping: bool,
) -> AsyncIterator[Result[T, Any]]:
    """Internal helper to wrap an asynchronous iterator with exception handling."""
    try:
        async for val in original_gen:
            yield Ok(val)
    except catch_tuple as e:
        yield Err(exc_map.get(type(e), e) if has_mapping else e)
    except Exception as e:  # noqa: BLE001
        tb = e.__traceback__
        raise e.with_traceback(tb.tb_next if tb else None) from None


def catch_each_iter[T_local, E_local: Exception, **P_local](
    exceptions: type[E_local] | tuple[type[E_local], ...] | Mapping[type[E_local], Any],
    *,
    map_to: Any = None,
) -> Callable[[Callable[P_local, Iterator[T_local]]], Callable[P_local, SafeStream[T_local, Any]]]:
    """Wrap a generator function to capture iteration-level exceptions into a SafeStream.

    This decorator ensures that if an exception is raised during the iteration
    of the generator, it is caught and yielded as an `Err` variant.

    Args:
        exceptions: One or more exception types to catch, or a mapping of
            exception types to error values.
        map_to: Optional constant value to use as the error if an exception
            matches (only used if `exceptions` is not a mapping).

    Returns:
        A decorator that transforms Generator[T] -> SafeStream[T, E].

    Examples:
        >>> # 1. Simple catch (returns the caught instance)
        >>> @catch_each_iter(ValueError)
        ... def pump(n):
        ...     for i in range(n):
        ...         if i == 2:
        ...             raise ValueError("fail")
        ...         yield i
        >>> list(pump(3))
        [Ok(0), Ok(1), Err(ValueError('fail'))]

    """
    exc_map = _resolve_mapping(exceptions, map_to)  # type: ignore[arg-type]
    catch_tuple = tuple(exc_map.keys())
    has_mapping = map_to is not None or isinstance(exceptions, Mapping)

    def decorator(f: Callable[P_local, Iterator[T_local]]) -> Any:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            __tracebackhide__ = True
            return SafeStream(_wrap_gen_sync(f(*args, **kwargs), catch_tuple, exc_map, has_mapping=has_mapping))

        return wrapper

    return cast("Any", decorator)


def catch_each_iter_async[T_local, E_local: Exception, **P_local](
    exceptions: type[E_local] | tuple[type[E_local], ...] | Mapping[type[E_local], Any],
    *,
    map_to: Any = None,
) -> Callable[[Callable[P_local, AsyncIterator[T_local]]], Callable[P_local, SafeStreamAsync[T_local, Any]]]:
    """Wrap an async generator function to capture iteration-level exceptions into a SafeStreamAsync.

    This is the asynchronous version of `@catch_each_iter`.

    Args:
        exceptions: One or more exception types to catch, or a mapping of
            exception types to error values.
        map_to: Optional constant value to use as the error if an exception
            matches (only used if `exceptions` is not a mapping).

    Returns:
        A decorator that transforms AsyncGenerator[T] -> SafeStreamAsync[T, E].

    """
    exc_map = _resolve_mapping(exceptions, map_to)  # type: ignore[arg-type]
    catch_tuple = tuple(exc_map.keys())
    has_mapping = map_to is not None or isinstance(exceptions, Mapping)

    def decorator(f: Callable[P_local, AsyncIterator[T_local]]) -> Any:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            __tracebackhide__ = True
            return SafeStreamAsync(_wrap_gen_async(f(*args, **kwargs), catch_tuple, exc_map, has_mapping=has_mapping))

        return wrapper

    return cast("Any", decorator)


def catch_boundary(
    exceptions: type[Exception] | tuple[type[Exception], ...] | Mapping[type[Exception], Any],
    *,
    map_to: Any = None,
) -> Callable[[T_cls], T_cls]:
    """Wrap all public methods of a class with the @catch decorator.

    This is an 'Entry Adapter' that allows lifting an entire external SDK or
    client class into the Result world in a single declaration.

    Args:
        exceptions: The exceptions to catch on all methods.
        map_to: Optional constant error value.

    Returns:
        A class decorator.

    Static Analysis Note (Type Erasure):
        Using this decorator causes **Type Erasure**. Most Python type checkers
        (Mypy, Pyright) cannot currently track that the return types of all
        methods have been transformed from `T` to `Result[T, E]`.

    """

    def decorator(cls: T_cls) -> T_cls:
        for name, method in inspect.getmembers(cls, predicate=inspect.isroutine):
            if name.startswith("_"):
                continue
            # Wrap the method with @catch using original parameters
            # Use Any to satisfy ty's overload resolution
            setattr(cls, name, catch(cast("Any", exceptions), map_to=map_to)(method))
        return cls

    return decorator


class _CatchInstanceProxy:
    """Internal proxy that wraps all method calls of an instance with @catch."""

    def __init__(
        self,
        obj: Any,
        exceptions: type[Exception] | tuple[type[Exception], ...] | Mapping[type[Exception], Any],
        map_to: Any = None,
    ) -> None:
        # Use object.__setattr__ to avoid infinite recursion with __getattr__
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_exceptions", exceptions)
        object.__setattr__(self, "_map_to", map_to)

    def __getattr__(self, name: str) -> Any:
        obj = object.__getattribute__(self, "_obj")
        attr = getattr(obj, name)
        exceptions = object.__getattribute__(self, "_exceptions")
        map_to = object.__getattribute__(self, "_map_to")

        if name.startswith("_"):
            return attr

        if inspect.isroutine(attr):
            # Bind the routine to the original object to ensure 'self' is passed
            # This is important for instance methods
            bound_method = attr.__get__(obj, obj.__class__) if hasattr(attr, "__get__") else attr  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportAttributeAccessIssue]
            return catch(exceptions, map_to=map_to)(bound_method)  # pyright: ignore[reportUnknownArgumentType]
        return None

    def __repr__(self) -> str:
        obj = object.__getattribute__(self, "_obj")
        return f"catch_instance({obj!r})"


@overload
def catch_instance[T_obj](
    obj: T_obj,
    exceptions: type[Exception] | tuple[type[Exception], ...],
    *,
    map_to: Any = None,
) -> T_obj: ...


@overload
def catch_instance[T_obj](
    obj: T_obj,
    exceptions: Mapping[type[Exception], Any],
) -> T_obj: ...


def catch_instance(
    obj: Any,
    exceptions: Any,
    *,
    map_to: Any = None,
) -> Any:
    """Wrap a specific object instance so all method calls return Results.

    Ideal for third-party objects returned from factories that you don't
    control the class of.

    Args:
        obj: The instance to wrap.
        exceptions: The exceptions to catch.
        map_to: Optional constant error value.

    Returns:
        A proxy object that behaves like the original but wraps methods in @catch.

    Static Analysis Note (Type Erasure):
        Using this proxy causes **Type Erasure**. Most Python type checkers
        will believe the returned object is of type `T_obj`.

    """
    # Cast to Any so the type checker thinks it's the original type
    return cast("Any", _CatchInstanceProxy(obj, exceptions, map_to))


class SafeStream[T, E](Iterable["Result[T, E]"]):
    """A wrapper around a fallible generator that provides functional transposition.

    SafeStream captures exceptions during iteration and converts them into Err variants.
    It provides methods to transpose the entire stream into a single Result or Outcome.

    Note:
        Like generators, a SafeStream can only be iterated once.

    """

    def __init__(self, gen: Iterator[Result[T, E]]) -> None:
        """Initialize a SafeStream with a Result-yielding iterator."""
        self._gen = gen
        self._consumed = False

    def __iter__(self) -> Iterator[Result[T, E]]:
        """Iterate over the stream.

        Raises:
            RuntimeError: If the stream is iterated more than once.

        """
        if self._consumed:
            msg = "SafeStream can only be iterated once"
            raise RuntimeError(msg)
        self._consumed = True
        yield from self._gen

    def to_result(self) -> Result[list[T], E]:
        """Transpose the stream into a single Result (All-or-Nothing).

        If any item in the stream is an Err, the first Err encountered is returned.
        Otherwise, returns Ok(list) containing all success values.

        """
        return combine(list(self))

    def to_outcome(self) -> Outcome[list[T], list[E]]:
        """Transpose the stream into a fault-tolerant Outcome (Partial Success).

        Collects all success values and all error values into a single Outcome.

        """
        oks, errs = partition(list(self))
        try:
            from .outcome import Outcome  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                r"Outcome is not available. use the \`result_pattern\` pip package and not just the result.py file"
            ) from None

        return Outcome(oks, errs or None)


class SafeStreamAsync[T, E](AsyncIterable["Result[T, E]"]):
    """Async version of SafeStream for fallible asynchronous generators."""

    def __init__(self, gen: AsyncIterator[Result[T, E]]) -> None:
        """Initialize a SafeStreamAsync with a Result-yielding async iterator."""
        self._gen = gen
        self._consumed = False

    async def __aiter__(self) -> AsyncIterator[Result[T, E]]:
        """Iterate over the async stream.

        Raises:
            RuntimeError: If the stream is iterated more than once.

        """
        if self._consumed:
            msg = "SafeStreamAsync can only be iterated once"
            raise RuntimeError(msg)
        self._consumed = True
        async for item in self._gen:
            yield item

    async def to_result(self) -> Result[list[T], E]:
        """Transpose the async stream into a single Result (All-or-Nothing)."""
        items = [res async for res in self]
        return combine(items)

    async def to_outcome(self) -> Outcome[list[T], list[E]]:
        """Transpose the async stream into a fault-tolerant Outcome (Partial Success)."""
        items = [res async for res in self]
        oks, errs = partition(items)
        try:
            from .outcome import Outcome  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                r"Outcome is not available. use the \`result_pattern\` pip package and not just the result.py file"
            ) from None

        return Outcome(oks, errs or None)

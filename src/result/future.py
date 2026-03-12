"""# Future: Experimental and Alpha features for Result Pattern.

This module houses features that are currently in testing or alpha stage.
These features are designed to handle iteration-level errors and provide
fault-tolerant streaming primitives.

Note:
    API stability is not guaranteed. These features may change or be removed
    in future versions without a major version bump.

"""

# pyright: reportPrivateUsage=false
# mypy: disable-error-code="no-any-return"

from __future__ import annotations

import sys
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Iterator, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, cast, overload

from .result import Err, Ok, OkErr, Result, _resolve_mapping, combine, partition

if TYPE_CHECKING:
    from .outcome import Outcome


def _raise_assertion_error(message: str) -> Any:
    """Internal helper to raise AssertionError and hide this frame from traceback."""
    __tracebackhide__ = True
    try:
        raise AssertionError(message)  # noqa: TRY301
    except AssertionError as e:
        tb = e.__traceback__
        raise e.with_traceback(tb.tb_next if tb else None) from None


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

        >>> # 2. Catch with map_to
        >>> @catch_each_iter(ValueError, map_to="error")
        ... def pump_mapped(n):
        ...     for i in range(n):
        ...         if i == 1:
        ...             raise ValueError
        ...         yield i
        >>> list(pump_mapped(2))
        [Ok(0), Err('error')]

        >>> # 3. Catch multiple with mapping
        >>> err_map = {ValueError: "val_err", TypeError: "type_err"}
        >>> @catch_each_iter(err_map)
        ... def pump_multi(x):
        ...     if x == 0:
        ...         raise ValueError
        ...     if x == 1:
        ...         raise TypeError
        ...     yield "ok"
        >>> list(pump_multi(0))
        [Err('val_err')]

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

    Examples:
        >>> @catch_each_iter_async(ValueError)
        ... async def async_pump(n):
        ...     for i in range(n):
        ...         if i == 1:
        ...             raise ValueError("async fail")
        ...         yield i
        >>> [res async for res in async_pump(2)]
        [Ok(0), Err(ValueError('async fail'))]

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


class AssertOk:
    """A context manager for asserting that Results must be Ok.

    It automatically monitors local variable assignments within the block.
    If any local variable is assigned an `Err` variant, it raises an
    `AssertionError` immediately (fail-fast).

    Note:
        The automatic scanning only works for the local scope where the
        `with assert_ok()` block is defined.

    """

    def __init__(self, message: str = "Result was Err") -> None:
        """Initialize the assert_ok context with a custom message."""
        self.message = message
        self._initial_locals: set[str] = set()
        self._old_trace: Any = None
        self._is_scanning: bool = False

    def __enter__(self) -> AssertOk:
        """Enter the assert_ok context and install the fail-fast tracer."""
        # Capture current locals to avoid re-triggering on existing variables
        frame = sys._getframe(1)  # noqa: SLF001
        self._initial_locals = set(frame.f_locals.keys())

        # Install trace function for fail-fast detection
        self._old_trace = sys.gettrace()
        sys.settrace(self._trace_callback)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the assert_ok context and uninstall the tracer."""
        sys.settrace(self._old_trace)

    def _trace_callback(self, frame: Any, event: str, _arg: Any) -> Any:
        """Trace function that scans locals for Err variants after each line."""
        __tracebackhide__ = True
        # Prevent re-entrancy issues if scanning logic itself triggers tracer
        if self._is_scanning:
            return self._trace_callback

        if event == "line":
            self._is_scanning = True
            try:
                # Scan current local variables
                for name, value in frame.f_locals.items():
                    # Only check variables that were added during this block
                    if name not in self._initial_locals:
                        match value:
                            case Err(error_val):  # pyright: ignore[reportUnknownVariableType]
                                # Trigger the error
                                error_str = cast("Any", error_val)
                                _raise_assertion_error(f"{self.message}: {error_str}")
                            case _:
                                pass
            finally:
                self._is_scanning = False
        return self._trace_callback

    def check[T, E](self, result: Result[T, E]) -> T:
        """Verify that a result is Ok within the context.

        Args:
            result: The Result to verify.

        Returns:
            The success value if Ok.

        Raises:
            AssertionError: If the result is an Err.

        """
        __tracebackhide__ = True
        match result:
            case Err(e):  # pyright: ignore[reportUnknownVariableType]
                err_val = cast("Any", e)
                return _raise_assertion_error(f"{self.message}: {err_val}")
            case Ok(v):
                return v


@overload
def assert_ok[T, E](result_or_message: Result[T, E]) -> T: ...


@overload
def assert_ok(result_or_message: str = "Result was Err") -> AssertOk: ...


def assert_ok(result_or_message: Any = "Result was Err") -> Any:
    """A dual-purpose utility for asserting that a Result must be Ok.

    Can be used as a standalone function or as a context manager.
    If a Result is an Err, it raises an AssertionError.

    Examples:
        >>> # 1. Functional usage
        >>> val = assert_ok(Ok(10))
        >>> # assert_ok(Err("fail")) # Raises AssertionError

        >>> # 2. Automatic context manager usage (Fail-fast)
        >>> with assert_ok("Critical operations"):
        ...     res = Ok(1)  # Fine
        ...     # res2 = Err("boom") # Raises AssertionError immediately

        >>> # 3. Explicit check usage
        >>> with assert_ok() as ctx:
        ...     val = ctx.check(Ok(42))

    """
    __tracebackhide__ = True
    if isinstance(result_or_message, OkErr):
        match result_or_message:
            case Err(e):  # pyright: ignore[reportUnknownVariableType]
                err_msg = cast("Any", e)
                return _raise_assertion_error(f"assert_ok failed: {err_msg}")
            case Ok(v):  # pyright: ignore[reportUnknownVariableType]
                return cast("Any", v)

    msg = str(result_or_message) if isinstance(result_or_message, str) else "Result was Err"
    return AssertOk(msg)


class SafeStream[T, E](Iterable["Result[T, E]"]):
    """A wrapper around a fallible generator that provides functional transposition.

    SafeStream captures exceptions during iteration and converts them into Err variants.
    It provides methods to transpose the entire stream into a single Result or Outcome.

    Note:
        Like generators, a SafeStream can only be iterated once.

    Attributes:
        _gen: The internal iterator yielding Results.
        _consumed: Whether the stream has already been iterated.

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

        Returns:
            The combined Result of the stream.

        Examples:
            >>> @catch_each_iter(ValueError)
            ... def gen(fail):
            ...     yield 1
            ...     if fail:
            ...         raise ValueError("fail")
            ...     yield 2
            >>> gen(fail=False).to_result()
            Ok([1, 2])
            >>> gen(fail=True).to_result()
            Err(ValueError('fail'))

        """
        return combine(list(self))

    def to_outcome(self) -> Outcome[list[T], list[E]]:
        """Transpose the stream into a fault-tolerant Outcome (Partial Success).

        Collects all success values and all error values into a single Outcome.

        Returns:
            An Outcome containing two lists: successes and errors.

        Examples:
            >>> @catch_each_iter(ValueError)
            ... def gen():
            ...     yield 1
            ...     raise ValueError("err")
            >>> out = gen().to_outcome()
            >>> out.value
            [1]
            >>> out.error
            [ValueError('err')]

        """
        oks, errs = partition(list(self))
        try:
            from .outcome import Outcome  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "Outcome is not available. use the `result_pattern` pip package and not just the result.py file"
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
        """Transpose the async stream into a single Result (All-or-Nothing).

        Returns:
            The combined Result of all yielded items.

        """
        items = [res async for res in self]
        return combine(items)

    async def to_outcome(self) -> Outcome[list[T], list[E]]:
        """Transpose the async stream into a fault-tolerant Outcome (Partial Success).

        Returns:
            A master Outcome containing all success items and all errors.

        """
        items = [res async for res in self]
        oks, errs = partition(items)
        try:
            from .outcome import Outcome  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "Outcome is not available. use the `result_pattern` pip package and not just the result.py file"
            ) from None

        return Outcome(oks, errs or None)

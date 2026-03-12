"""# Future: Experimental and Alpha features for Result Pattern.

This module houses features that are currently in testing or alpha stage.
These features are designed to handle iteration-level errors and provide
fault-tolerant streaming primitives.

Note:
    API stability is not guaranteed. These features may change or be removed
    in future versions without a major version bump.

"""

# pyright: reportPrivateUsage=false
# mypy: disable-error-code="no-any-return, redundant-cast"

from __future__ import annotations

import asyncio
import inspect
import random
import sys
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from .result import Err, Ok, OkErr, Result, _resolve_mapping, catch, combine, partition
from .result import catch as _catch_decorator

if TYPE_CHECKING:
    from .outcome import Outcome

T_cls = TypeVar("T_cls", bound=type[Any])
T_obj = TypeVar("T_obj")


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
        - Your IDE may still show the original return types.
        - You may need to use `Any` or explicit type stubs when calling
          decorated methods to avoid false-positive type errors.

    Examples:
        >>> @catch_boundary(ValueError, map_to="domain_error")
        ... class Client:
        ...     def perform(self, x):
        ...         if x < 0:
        ...             raise ValueError
        ...         return x
        >>> Client().perform(-1)
        Err('domain_error')

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
        return attr

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
        will believe the returned object is of type `T_obj` (with original
        return types), but at runtime every method will return a `Result`.
        - You may need to cast the result to `Any` or use `# type: ignore`
          when calling methods on the proxy to satisfy the type checker.

    Examples:
        >>> class Raw:
        ...     def run(self):
        ...         raise ValueError("fail")
        >>> safe = catch_instance(Raw(), ValueError)
        >>> safe.run()
        Err(ValueError('fail'))

    """
    # Cast to Any so the type checker thinks it's the original type
    return cast("Any", _CatchInstanceProxy(obj, exceptions, map_to))


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

    Functional Mode:
        When passed a `Result`, it returns the success value or raises
        `AssertionError` immediately. This is the high-performance way to
        assert invariants.

    Context Manager Mode:
        When used as a context manager, it automatically monitors local variable
        assignments within the block using `sys.settrace`. If any local variable
        is assigned an `Err` variant, it raises an `AssertionError` (fail-fast).

    Performance & Behavior Notes:
        - **Overhead**: The context manager installs a trace function, which
          introduces performance overhead compared to functional
          usage. Use it for scripts and prototypes rather than hot loops.
        - **Scanning Scope**: The automatic scanning only catches `Err` variants
          that are **assigned to a variable** name in the local scope.
          Unassigned return values will NOT be caught.

    Examples:
        >>> # 1. Functional usage (Low overhead)
        >>> val = assert_ok(Ok(10))
        >>> # assert_ok(Err("fail")) # Raises AssertionError

        >>> # 2. Automatic context manager usage (Higher overhead, Fail-fast)
        >>> with assert_ok("Critical operations"):
        ...     res = Ok(1)  # Fine
        ...     # res2 = Err("boom") # Raises AssertionError immediately

        >>> # 3. Explicit check usage (Lower overhead in context)
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


def _get_retry_delay(
    current_delay: float,
    *,
    jitter: bool | float,
) -> float:
    """Internal helper to calculate next retry delay."""
    if current_delay <= 0:
        return 0
    sleep_time = current_delay
    if jitter:
        # If jitter is True, default to 0.1, else use the provided float value
        jitter_val = jitter if isinstance(jitter, float) else 0.1
        sleep_time += random.uniform(0, jitter_val)
    return sleep_time


@overload
def retry_result[T, E, **P](
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    retry_if: Callable[[E], bool],
    catch: None = None,
) -> Callable[[Callable[P, Result[T, E]]], Callable[P, Result[T, E]]]: ...


@overload
def retry_result[T, E, **P](
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    retry_if: None = None,
    catch: None = None,
) -> Callable[[Callable[P, Result[T, E]]], Callable[P, Result[T, E]]]: ...


@overload
def retry_result[T, E_exc: Exception, **P](
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    retry_if: Callable[[E_exc], bool] | None = None,
    catch: type[E_exc],
) -> Callable[[Callable[P, T]], Callable[P, Result[T, E_exc]]]: ...


@overload
def retry_result[T, **P](
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    catch: tuple[type[Exception], ...] | Mapping[type[Exception], Any],
) -> Callable[[Callable[P, T | Result[T, Any]]], Callable[P, Result[T, Any]]]: ...


def retry_result(
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    retry_if: Callable[[Any], bool] | None = None,
    catch: type[Exception] | tuple[type[Exception], ...] | Mapping[type[Exception], Any] | None = None,
) -> Any:
    """A resilience decorator for synchronous functions that return a `Result`.

    It will automatically re-execute the function if it returns an `Err` variant,
    up to the specified number of `attempts`.

    If `catch` is provided, it will also catch specified exceptions and turn
    them into `Err` variants before deciding whether to retry.

    Args:
        attempts: Total number of attempts to try (default 3).
        delay: Initial delay in seconds between retries (default 0).
        backoff: Multiplier for the delay after each attempt (default 1.0).
        jitter: If True (or a float), add randomness to the delay.
        retry_if: Optional predicate to decide if an error should be retried.
        catch: Optional exception types to catch and lift into Results.

    Returns:
        A decorator that adds retry logic to the function.

    Notes & Footguns:
        - **Idempotency**: Retrying functions with side effects (e.g., DB writes)
          can be dangerous if the operation is not idempotent.
        - **Exception Scoping**: If `catch` is NOT used, and the function raises
          an exception, the retry logic will NOT trigger (the exception will
          bubble up). The retry logic only reacts to `Err` return values.
        - **Wait Times**: High `attempts` and `backoff` values can lead to
          extremely long execution times.

    Examples:
        >>> # 1. Basic Retry (Reacts to Err return)
        >>> @retry_result(attempts=3)
        ... def unstable():
        ...     return Err("fail")
        >>> unstable()
        Err('fail')  # Tried 3 times

        >>> # 2. Exponential Backoff with Jitter
        >>> @retry_result(attempts=5, delay=0.1, backoff=2.0, jitter=True)
        ... def network_call():
        ...     return Err("timeout")

        >>> # 3. Conditional Retry (only retry on transient errors)
        >>> @retry_result(attempts=3, retry_if=lambda e: e == "temporary")
        ... def db_op():
        ...     return Err("permanent")
        >>> db_op()
        Err('permanent')  # Fails fast, only tried once

        >>> # 4. Internal Exception Catching (Lifting)
        >>> @retry_result(attempts=3, catch=ValueError)
        ... def parse_stuff(s):
        ...     return int(s)  # Raises ValueError -> Err -> Retry
        >>> parse_stuff("abc")
        Err(ValueError("invalid literal for int()..."))

        >>> # 5. Stacking with @catch
        >>> @retry_result(attempts=2)
        ... @catch(KeyError)
        ... def get_config():
        ...     raise KeyError("missing")
        >>> get_config()
        Err(KeyError('missing'))  # @catch fires first, then @retry sees Err

    """

    def decorator[T, E, **P](f: Callable[P, Result[T, E] | T]) -> Callable[P, Result[T, E]]:
        # If catch is provided, we wrap the function in @catch first
        wrapped_f = _catch_decorator(cast("Any", catch))(f) if catch else f

        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Result[T, E]:
            current_delay = delay
            # Initialize with dummy error, will be overwritten in loop
            res: Any = Err(cast("E", "retry loop didn't run"))  # pyright: ignore[reportUnknownVariableType]
            for i in range(attempts):
                # Call the (potentially catch-wrapped) function
                res = wrapped_f(*args, **kwargs)

                # If it's not a Result (e.g. catch wasn't used and func returns T),
                # we wrap it in Ok automatically.
                if not isinstance(res, OkErr):
                    res = Ok(res)

                if isinstance(res, Ok):
                    return res  # pyright: ignore[reportUnknownVariableType]

                # It's an Err, check if we should retry
                err_val = res.err()  # pyright: ignore[reportUnknownVariableType]
                if retry_if and not retry_if(err_val):
                    return res  # pyright: ignore[reportUnknownVariableType]

                # Final attempt failed
                if i == attempts - 1:
                    return res  # pyright: ignore[reportUnknownVariableType]

                # Sleep and backoff
                sleep_time = _get_retry_delay(current_delay, jitter=jitter)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                current_delay *= backoff
            return res  # pyright: ignore[reportUnknownVariableType]

        return cast("Any", wrapper)

    return cast("Any", decorator)


@overload
def retry_result_async[T, E, **P](
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    retry_if: Callable[[E], bool],
    catch: None = None,
) -> Callable[[Callable[P, Awaitable[Result[T, E]]]], Callable[P, Awaitable[Result[T, E]]]]: ...


@overload
def retry_result_async[T, E, **P](
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    retry_if: None = None,
    catch: None = None,
) -> Callable[[Callable[P, Awaitable[Result[T, E]]]], Callable[P, Awaitable[Result[T, E]]]]: ...


@overload
def retry_result_async[T, E_exc: Exception, **P](
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    retry_if: Callable[[E_exc], bool] | None = None,
    catch: type[E_exc],
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[Result[T, E_exc]]]]: ...


@overload
def retry_result_async[T, **P](
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    catch: tuple[type[Exception], ...] | Mapping[type[Exception], Any],
) -> Callable[[Callable[P, Awaitable[T] | Awaitable[Result[T, Any]]]], Callable[P, Awaitable[Result[T, Any]]]]: ...


def retry_result_async(
    attempts: int = 3,
    delay: float = 0,
    backoff: float = 1.0,
    *,
    jitter: bool | float = False,
    retry_if: Callable[[Any], bool] | None = None,
    catch: type[Exception] | tuple[type[Exception], ...] | Mapping[type[Exception], Any] | None = None,
) -> Any:
    """A resilience decorator for asynchronous functions that return a `Result`.

    This is the asynchronous version of `retry_result`.

    It will automatically re-execute the function if it returns an `Err` variant,
    up to the specified number of `attempts`.

    If `catch` is provided, it will also catch specified exceptions and turn
    them into `Err` variants before deciding whether to retry.

    Args:
        attempts: Total number of attempts to try (default 3).
        delay: Initial delay in seconds between retries (default 0).
        backoff: Multiplier for the delay after each attempt (default 1.0).
        jitter: If True (or a float), add randomness to the delay.
        retry_if: Optional predicate to decide if an error should be retried.
        catch: Optional exception types to catch and lift into Results.

    Returns:
        A decorator that adds retry logic to the async function.

    Notes & Footguns:
        - **Idempotency**: Retrying functions with side effects (e.g., POST calls)
          can be dangerous if the operation is not idempotent.
        - **Exception Scoping**: If `catch` is NOT used, and the function raises
          an exception, the retry logic will NOT trigger (the exception will
          bubble up).
        - **Concurrency**: This decorator retries in series. For parallel
          retries, consider other orchestration patterns.

    Examples:
        >>> # 1. Async Basic Retry
        >>> @retry_result_async(attempts=3)
        ... async def unstable():
        ...     return Err("fail")
        >>> await unstable()
        Err('fail')

        >>> # 2. Async Backoff with Jitter
        >>> @retry_result_async(attempts=5, delay=0.1, backoff=2.0, jitter=0.5)
        ... async def fetch_data():
        ...     return Err("timeout")

        >>> # 3. Async Conditional Retry
        >>> @retry_result_async(attempts=3, retry_if=lambda e: e.status == 429)
        ... async def api_call():
        ...     return Err(Response(status=500))
        >>> await api_call()
        Err(Response(status=500))  # Fails fast, not a 429

        >>> # 4. Async Internal Exception Catching
        >>> @retry_result_async(attempts=3, catch=asyncio.TimeoutError)
        ... async def timed_op():
        ...     raise asyncio.TimeoutError
        >>> await timed_op()
        Err(asyncio.TimeoutError())

    """

    def decorator[T, E, **P](
        f: Callable[P, Awaitable[Result[T, E]] | Awaitable[T]],
    ) -> Callable[P, Awaitable[Result[T, E]]]:
        # If catch is provided, we wrap the function in @catch first
        # We must ensure the resulting wrapper is awaited
        wrapped_f = _catch_decorator(cast("Any", catch))(f) if catch else f

        @wraps(f)
        async def wrapper(*args: Any, **kwargs: Any) -> Result[T, E]:
            current_delay = delay
            res: Any = Err(cast("E", "retry loop didn't run"))  # pyright: ignore[reportUnknownVariableType]
            for i in range(attempts):
                res = await wrapped_f(*args, **kwargs)  # pyright: ignore[reportUnknownVariableType, reportGeneralTypeIssues]

                if not isinstance(res, OkErr):
                    res = Ok(res)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

                if isinstance(res, Ok):
                    return res  # pyright: ignore[reportUnknownVariableType, reportUnknownVariableType]

                err_val = res.err()  # pyright: ignore[reportUnknownVariableType]
                if retry_if and not retry_if(err_val):
                    return res  # pyright: ignore[reportUnknownVariableType]

                if i == attempts - 1:
                    return res  # pyright: ignore[reportUnknownVariableType]

                sleep_time = _get_retry_delay(current_delay, jitter=jitter)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                current_delay *= backoff
            return res  # pyright: ignore[reportUnknownVariableType]

        return cast("Any", wrapper)

    return cast("Any", decorator)

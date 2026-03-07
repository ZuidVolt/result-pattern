"""# Result: Functional Error Handling for Modern Python.

A lightweight, single-file library designed to implement the 'Errors as Values'
pattern in Python 3.12+. This library tries to help bridge the gap between pure
functional safety and the pragmatic realities of the exception-heavy Python ecosystem.

## Core Philosophy

In standard Python, errors are implicit side-effects (Exceptions). In this
library, errors are explicit return values. This forcing function ensures
that failure paths are handled as diligently as success paths.

## Core Goals

1.  **Explicit Error Handling**: Shift from 'invisible' exceptions to explicit
    `Result` return types, making failure modes a first-class part of the API.
2.  **Reduced Cognitive Load**: Use the `@do` and `@do_async` decorators to
    write linear, procedural-looking code that automatically handles
    short-circuiting logic.
3.  **Static Analysis First**: Leverage modern Python typing features (PEP 695,
    PEP 742) to provide perfect type narrowing and static verification in
    tools like Basedpyright, Mypy and Ty.
4.  **Zero-Escape Safety**: Isolate crashing operations (panics) within the
    `.unsafe` namespace to ensure that 'unwrapping' is always an intentional,
    visible choice.
5.  **Pragmatic Interop**: Provide 'lifting' tools like `@safe` to seamlessly
    convert standard Python exception-throwing code into functional containers.

## Limitations & Constraints

*   **Type Invariance**: Python's type system can be rigid regarding variance
    in Unions. Returning a specific `Err(ValueError)` when a signature expects
    `Result[T, Exception]` may occasionally require explicit type annotations.
*   **Higher-Kinded Types**: Python lacks native support for higher-kinded
    types, making operations like `flatten()` return `Result[Any, Any]` to
    maintain runtime flexibility while sacrificing some static detail.
*   **Ecosystem Inertia**: External Python libraries will continue to raise
    exceptions. Lifting must be performed at the integration boundaries.
"""

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, NoReturn, TypeIs

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Generator, Iterable

# --- Exceptions ---


class UnwrapError(RuntimeError):
    """Raised when unwrapping a Result variant that does not contain the expected value.

    This exception signals a 'panic' state where the developer made an assumption
    about the Result state that was incorrect at runtime.

    Attributes:
        result: The original Result instance (Ok or Err) that caused the panic.
            Allowing for post-mortem inspection of the failed state.
    """

    def __init__(self, result: Result[Any, Any], message: str) -> None:
        """Initialize the panic with context.

        Args:
            result: The instance that failed to unwrap.
            message: Descriptive error message explaining why the unwrap failed.
        """
        self.result = result
        super().__init__(message)


class _DoError(Exception):
    """Internal exception for do-notation flow control."""

    def __init__(self, err: Err[Any]) -> None:
        self.err = err


# --- Type Aliases ---

type Result[T, E] = Ok[T] | Err[E]
"""
A type that represents either success (`Ok`) or failure (`Err`).

This is the foundational type for explicit error handling. Use `is_ok()` and
`is_err()` to narrow the type or `match` for exhaustive handling.

Examples:
    >>> def get_user(id: int) -> Result[dict, str]:
    ...     return Ok({"id": id}) if id > 0 else Err("Invalid ID")
"""

type Do[T, E] = Generator[Result[Any, E], Any, T]
"""
A type alias for generator functions compatible with the `@do` decorator.

The generator yields `Result` variants to unwrap them and eventually returns
a value of type `T`.
"""

type DoAsync[T, E] = AsyncGenerator[Result[Any, E], Any]
"""
A type alias for async generator functions compatible with the `@do_async` decorator.

Async generators must yield an `Ok` variant as their final step to simulate
a return value.
"""


# --- Unsafe Proxies ---


@dataclass(frozen=True, slots=True)
class _OkUnsafe[T]:
    """Namespace for operations that may raise an exception on an Ok variant."""

    _owner: Ok[T]

    def unwrap(self) -> T:
        """Extract the contained value. Always succeeds on Ok.

        Returns:
            The contained success value.
        """
        return self._owner._value  # pyright: ignore[reportPrivateUsage] # noqa: SLF001

    def unwrap_err(self) -> NoReturn:
        """Raise an UnwrapError because an Ok value does not contain an error.

        Raises:
            UnwrapError: Always raised with a descriptive message.
        """
        msg = f"Called unwrap_err on an Ok value: {self._owner._value!r}"  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        raise UnwrapError(self._owner, msg)

    def expect(self, _msg: str) -> T:
        """Extract the contained value, ignoring the custom message.

        Always succeeds on `Ok` variants.

        Args:
            _msg: The custom panic message (ignored on success).

        Returns:
            The contained success value.
        """
        return self._owner._value  # pyright: ignore[reportPrivateUsage] # noqa: SLF001

    def unwrap_or_raise(self, _e: type[Exception]) -> T:
        """Extract the contained value, ignoring the exception type.

        Always succeeds on `Ok` variants.

        Args:
            _e: The exception type to raise (ignored on success).

        Returns:
            The contained success value.
        """
        return self._owner._value  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


@dataclass(frozen=True, slots=True)
class _ErrUnsafe[E]:
    """Namespace for operations that may raise an exception on an Err variant."""

    _owner: Err[E]

    def unwrap(self) -> NoReturn:
        """Raise an UnwrapError because an error cannot be unwrapped.

        Raises:
            UnwrapError: Always raised, containing the error state.
        """
        msg = f"Called unwrap on an Err value: {self._owner._error!r}"  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        raise UnwrapError(self._owner, msg)

    def unwrap_err(self) -> E:
        """Extract the contained error. Always succeeds on Err.

        Returns:
            The contained error state.
        """
        return self._owner._error  # pyright: ignore[reportPrivateUsage] # noqa: SLF001

    def expect(self, msg: str) -> NoReturn:
        """Raise an UnwrapError with a custom message.

        Args:
            msg: The custom message to prepend to the error state output.

        Raises:
            UnwrapError: Always raised.
        """
        msg_final = f"{msg}: {self._owner._error!r}"  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        raise UnwrapError(self._owner, msg_final)

    def unwrap_or_raise(self, e: type[Exception]) -> NoReturn:
        """Raise a custom exception with the error state.

        Args:
            e: The exception class to instantiate and raise.

        Raises:
            Exception: An instance of `e` initialized with the error state.
        """
        raise e(self._owner._error)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001


# --- Core Variants ---


@dataclass(frozen=True, slots=True)  # noqa: PLR0904
class Ok[T]:
    """A container representing a successful computation result.

    To access the value safely, use pattern matching, functional chaining,
    or the `.ok()` conversion method.

    Examples:
        >>> res = Ok(200)
        >>> match res:
        ...     case Ok(v):
        ...         print(f"Success: {v}")
        Success: 200
    """

    _value: T
    __match_args__ = ("_value",)

    def __init__(self, value: T) -> None:
        """Initialize an Ok variant.

        Args:
            value: The successful result to wrap.
        """
        object.__setattr__(self, "_value", value)

    def __iter__(self) -> Generator[Any, Any, T]:
        """Allow use in generator expressions for do-notation."""
        return (yield self._value)  # noqa: B901

    async def __aiter__(self) -> AsyncGenerator[T, Any]:
        """Allow use in async generator expressions for do-notation."""
        yield self._value

    @property
    def unsafe(self) -> _OkUnsafe[T]:
        """Namespace for operations that might panic (raise exceptions).

        Examples:
            >>> Ok(10).unsafe.unwrap()
            10
        """
        return _OkUnsafe(self)

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"

    def is_ok(self) -> bool:
        """Check if the result is an Ok variant.

        Returns:
            Always True for Ok instances.
        """
        return True

    def is_err(self) -> bool:
        """Check if the result is an Err variant.

        Returns:
            Always False for Ok instances.
        """
        return False

    def __getattr__(self, name: str) -> NoReturn:
        """Educational runtime safeguard against common API mistakes."""
        if name in {
            "value",
            "error",
            "unwrap",
            "unwrap_err",
            "expect",
            "expect_err",
            "unwrap_or_raise",
            "inspect",
            "inspect_async",
            "inspect_err",
        }:
            _raise_api_error(name)
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    # --- error for incorrect API access ---

    def unwrap(self) -> NoReturn:
        """Root-level unwrap is disabled. Use .unsafe.unwrap() instead."""
        _raise_api_error("unwrap")

    def unwrap_err(self) -> NoReturn:
        """Root-level unwrap_err is disabled. Use .unsafe.unwrap_err() instead."""
        _raise_api_error("unwrap_err")

    def expect(self, _msg: str) -> NoReturn:
        """Root-level expect is disabled. Use .unsafe.expect() instead."""
        _raise_api_error("expect")

    def unwrap_or_raise(self, _e: type[Exception]) -> NoReturn:
        """Root-level unwrap_or_raise is disabled. Use .unsafe.unwrap_or_raise() instead."""
        _raise_api_error("unwrap_or_raise")

    # --- Functional API ---

    def map[U](self, func: Callable[[T], U]) -> Ok[U]:
        """Apply a function to the contained value.

        Args:
            func: A pure function to transform the success value.

        Returns:
            A new `Ok` containing the transformed value.

        Examples:
            >>> Ok(10).map(lambda x: x * 2)
            Ok(20)
        """
        return Ok(func(self._value))

    async def map_async[U](self, func: Callable[[T], Awaitable[U]]) -> Ok[U]:
        """Apply an async function to the contained value.

        Args:
            func: An async function to transform the success value.

        Returns:
            A new Ok containing the transformed value.

        Examples:
            >>> async def double(x):
            ...     return x * 2
            >>> await Ok(10).map_async(double)
            Ok(20)
        """
        return Ok(await func(self._value))

    def map_or[U](self, _default: U, func: Callable[[T], U]) -> U:
        """Apply a function to the value or return a default.

        Args:
            _default: The fallback value used if this were an Err.
            func: A function to transform the success value.

        Returns:
            The transformed value.

        Examples:
            >>> Ok(10).map_or(0, lambda x: x * 2)
            20
        """
        return func(self._value)

    def map_or_else[U](self, _default_func: Callable[[], U], func: Callable[[T], U]) -> U:
        """Apply a function to the value or compute a default.

        Args:
            _default_func: A function to generate a fallback value.
            func: A function to transform the success value.

        Returns:
            The transformed value.

        Examples:
            >>> Ok(10).map_or_else(lambda: 0, lambda x: x * 2)
            20
        """
        return func(self._value)

    def map_err(self, _func: Callable[[Any], Any]) -> Ok[T]:
        """Ignore the error mapping and return self unchanged."""
        return self

    def replace[U](self, value: U) -> Ok[U]:
        """Discard the contained value and replace it with a constant.

        Args:
            value: The new value to wrap in Ok.

        Examples:
            >>> Ok(10).replace("done")
            Ok('done')
        """
        return Ok(value)

    def replace_err(self, _error: object) -> Ok[T]:
        """Ignore the error replacement and return self unchanged."""
        return self

    def tap(self, func: Callable[[T], Any]) -> Ok[T]:
        """Call a function with the contained value for side effects.

        Args:
            func: A function called with the success value.

        Returns:
            The current instance unchanged.

        Examples:
            >>> Ok(10).tap(print).map(lambda x: x + 1)
            10
            Ok(11)
        """
        func(self._value)
        return self

    async def tap_async(self, func: Callable[[T], Awaitable[Any]]) -> Ok[T]:
        """Call an async function with the contained value for side effects.

        Args:
            func: An async function called with the success value.

        Returns:
            The current instance unchanged.

        Examples:
            >>> async def log(x):
            ...     print(f"Logging: {x}")
            >>> await Ok(10).tap_async(log)
            Logging: 10
            Ok(10)
        """
        await func(self._value)
        return self

    def tap_err(self, _func: Callable[[Any], Any]) -> Ok[T]:
        """Ignore the error side effect and return self unchanged."""
        return self

    def inspect(self, _func: Callable[[T], Any]) -> NoReturn:
        """Root-level inspect is disabled. Use .tap() instead."""
        _raise_api_error("inspect")

    def inspect_async(self, _func: Callable[[T], Awaitable[Any]]) -> NoReturn:
        """Root-level inspect_async is disabled. Use .tap_async() instead."""
        _raise_api_error("inspect_async")

    def inspect_err(self, _func: Callable[[Any], Any]) -> NoReturn:
        """Root-level inspect_err is disabled. Use .tap_err() instead."""
        _raise_api_error("inspect_err")

    def and_then[U, E_local](self, func: Callable[[T], Result[U, E_local]]) -> Result[U, E_local]:
        """Chain another result-returning operation (FlatMap).

        Args:
            func: A function that takes the value and returns a new `Result`.

        Returns:
            The `Result` returned by `func`.

        Examples:
            >>> def validate(n: int) -> Result[int, str]:
            ...     return Ok(n) if n > 0 else Err("too small")
            >>> Ok(10).and_then(validate)
            Ok(10)
        """
        return func(self._value)

    async def and_then_async[U, E_local](
        self, func: Callable[[T], Awaitable[Result[U, E_local]]]
    ) -> Result[U, E_local]:
        """Chain an async computation that might fail.

        Args:
            func: An async function that takes the value and returns a new Result.

        Returns:
            The Result returned by func.

        Examples:
            >>> async def check(n):
            ...     return Ok(n) if n > 0 else Err("low")
            >>> await Ok(10).and_then_async(check)
            Ok(10)
        """
        return await func(self._value)

    def or_else(self, _func: Callable[[Any], Result[T, Any]]) -> Ok[T]:
        """Ignore the recovery function and return self unchanged."""
        return self

    def flatten(self) -> Result[Any, Any]:
        """Flatten a nested Result.

        If the contained value is itself a `Result` (Ok or Err), it is returned.
        Otherwise, the current instance is returned unchanged.

        Returns:
            The inner Result or self.

        Examples:
            >>> Ok(Ok(10)).flatten()
            Ok(10)
            >>> Ok(Err("fail")).flatten()
            Err('fail')
        """
        val = self._value  # pyright: ignore[reportUnknownMemberType]
        if isinstance(val, Ok | Err):
            return val  # pyright: ignore[reportUnknownVariableType]
        return self

    def filter[E_local](self, predicate: Callable[[T], bool], error: E_local) -> Result[T, E_local]:
        """Convert success to failure if a condition is not met.

        Args:
            predicate: A function that returns True for valid values.
            error: The error state to use if the predicate returns False.

        Returns:
            Self if predicate is True, otherwise `Err(error)`.

        Examples:
            >>> Ok(10).filter(lambda x: x > 5, "error")
            Ok(10)
            >>> Ok(3).filter(lambda x: x > 5, "too small")
            Err('too small')
        """
        if predicate(self._value):
            return self
        return Err(error)

    def match[U](self, on_ok: Callable[[T], U], on_err: Callable[[Any], U]) -> U:  # noqa: ARG002
        """Exhaustively handle both success and failure cases.

        Args:
            on_ok: Function to call with the success value.
            on_err: Function to call with the error state.

        Returns:
            The result of whichever function was called.

        Examples:
            >>> res = Ok(10)
            >>> res.match(on_ok=lambda x: x * 2, on_err=lambda e: 0)
            20
        """
        return on_ok(self._value)

    def unwrap_or(self, _default: object) -> T:
        """Extract the contained value, ignoring the default.

        Args:
            _default: Ignored fallback value.

        Returns:
            The contained success value.

        Examples:
            >>> Ok(10).unwrap_or(0)
            10
        """
        return self._value

    def unwrap_or_else(self, _func: Callable[[Any], T]) -> T:
        """Extract the contained value, ignoring the fallback function.

        Args:
            _func: Ignored fallback function.

        Returns:
            The contained success value.

        Examples:
            >>> Ok(10).unwrap_or_else(lambda e: 0)
            10
        """
        return self._value

    def ok(self) -> T:
        """Convert to Optional[T].

        Returns:
            The contained value. Always succeeds on `Ok`.

        Examples:
            >>> Ok(10).ok()
            10
        """
        return self._value

    def err(self) -> None:
        """Convert to Optional[E].

        Returns:
            Always returns `None` for `Ok` variants.

        Examples:
            >>> Ok(10).err()
            None
        """
        return


@dataclass(frozen=True, slots=True)  # noqa: PLR0904
class Err[E]:
    """A failed result containing an error state.

    To handle the error safely, use pattern matching or recovery methods like
    `.or_else()` or `.unwrap_or()`.

    Examples:
        >>> res = Err("not found")
        >>> match res:
        ...     case Err(e):
        ...         print(f"Error: {e}")
        Error: not found
    """

    _error: E
    __match_args__ = ("_error",)

    def __init__(self, error: E) -> None:
        """Initialize an Err variant.

        Args:
            error: The error state or exception to wrap.
        """
        object.__setattr__(self, "_error", error)

    def __iter__(self) -> Generator[Result[Any, E], Any, NoReturn]:
        """Allow use in generator expressions for do-notation."""
        raise _DoError(self)
        yield self  # should be UNREACHABLE

    async def __aiter__(self) -> AsyncGenerator[NoReturn, Any]:
        """Allow use in async generator expressions for do-notation."""
        raise _DoError(self)
        yield self  # should be UNREACHABLE

    @property
    def unsafe(self) -> _ErrUnsafe[E]:
        """Namespace for operations that might panic.

        Examples:
            >>> Err("fail").unsafe.unwrap_err()
            'fail'
        """
        return _ErrUnsafe(self)

    def __repr__(self) -> str:
        return f"Err({self._error!r})"

    def is_ok(self) -> bool:
        """Check if the result is an Ok variant.

        Returns:
            Always False for Err instances.
        """
        return False

    def is_err(self) -> bool:
        """Check if the result is an Err variant.

        Returns:
            Always True for Err instances.
        """
        return True

    def __getattr__(self, name: str) -> NoReturn:
        """Educational runtime safeguard against common API mistakes."""
        if name in {
            "value",
            "error",
            "unwrap",
            "unwrap_err",
            "expect",
            "expect_err",
            "unwrap_or_raise",
            "inspect",
            "inspect_async",
            "inspect_err",
        }:
            _raise_api_error(name)
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    # --- error for incorrect API access ---

    def unwrap(self) -> NoReturn:
        """Root-level unwrap is disabled. Use .unsafe.unwrap() instead."""
        _raise_api_error("unwrap")

    def unwrap_err(self) -> NoReturn:
        """Root-level unwrap_err is disabled. Use .unsafe.unwrap_err() instead."""
        _raise_api_error("unwrap_err")

    def expect(self, _msg: str) -> NoReturn:
        """Root-level expect is disabled. Use .unsafe.expect() instead."""
        _raise_api_error("expect")

    def unwrap_or_raise(self, _e: type[Exception]) -> NoReturn:
        """Root-level unwrap_or_raise is disabled. Use .unsafe.unwrap_or_raise() instead."""
        _raise_api_error("unwrap_or_raise")

    # --- Functional API ---

    def map[U, T_local](self, _func: Callable[[T_local], U]) -> Err[E]:
        """Ignore the value mapping and return self unchanged."""
        return self

    async def map_async[U, T_local](self, _func: Callable[[T_local], Awaitable[U]]) -> Err[E]:
        """Ignore success mapping and return self unchanged.

        Args:
            _func: Ignored async mapping function.

        Returns:
            The current instance unchanged.
        """
        return self

    def map_or[U](self, default: U, _func: object) -> U:
        """Ignore the mapping and return the default.

        Args:
            default: The fallback value.
            _func: Ignored mapping function.

        Returns:
            The default value.

        Examples:
            >>> Err("fail").map_or(0, lambda x: x * 2)
            0
        """
        return default

    def map_or_else[U](self, default_func: Callable[[], U], _func: object) -> U:
        """Ignore the mapping and compute the default.

        Args:
            default_func: Function to generate the fallback value.
            _func: Ignored mapping function.

        Returns:
            The computed default value.

        Examples:
            >>> Err("fail").map_or_else(lambda: 0, lambda x: x * 2)
            0
        """
        return default_func()

    def map_err[F](self, func: Callable[[E], F]) -> Err[F]:
        """Apply a function to the contained error.

        Args:
            func: A function to transform the error state.

        Returns:
            A new `Err` containing the transformed error.

        Examples:
            >>> Err(404).map_err(lambda code: f"Code: {code}")
            Err('Code: 404')
        """
        return Err(func(self._error))

    def replace(self, _value: object) -> Err[E]:
        """Ignore the replacement and return self unchanged."""
        return self

    def replace_err[F](self, error: F) -> Err[F]:
        """Discard the contained error and replace it with a constant.

        Args:
            error: The new error to wrap in Err.

        Returns:
            A new `Err` instance.

        Examples:
            >>> Err("timeout").replace_err("network error")
            Err('network error')
        """
        return Err(error)

    def tap(self, _func: Callable[[Any], Any]) -> Err[E]:
        """Ignore the side effect and return self unchanged."""
        return self

    async def tap_async[T_local](self, _func: Callable[[T_local], Awaitable[Any]]) -> Err[E]:
        """Ignore async success side effects.

        Args:
            _func: Ignored async side effect function.

        Returns:
            The current instance unchanged.
        """
        return self

    def tap_err(self, func: Callable[[E], Any]) -> Err[E]:
        """Call a function with the contained error for side effects.

        Args:
            func: A function called with the error state.

        Returns:
            The current instance unchanged.

        Examples:
            >>> Err("db fail").tap_err(print)
            db fail
            Err('db fail')
        """
        func(self._error)
        return self

    def inspect(self, _func: Callable[[Any], Any]) -> NoReturn:
        """Root-level inspect is disabled. Use .tap() instead."""
        _raise_api_error("inspect")

    def inspect_async(self, _func: Callable[[Any], Awaitable[Any]]) -> NoReturn:
        """Root-level inspect_async is disabled. Use .tap_async() instead."""
        _raise_api_error("inspect_async")

    def inspect_err(self, _func: Callable[[E], Any]) -> NoReturn:
        """Root-level inspect_err is disabled. Use .tap_err() instead."""
        _raise_api_error("inspect_err")

    def and_then(self, _func: Callable[[Any], Result[Any, Any]]) -> Err[E]:
        """Short-circuit the chain and return self unchanged."""
        return self

    async def and_then_async[U, T_local](self, _func: Callable[[T_local], Awaitable[Result[U, E]]]) -> Err[E]:
        """Short-circuit async success chaining.

        Args:
            _func: Ignored async success chaining function.

        Returns:
            The current instance unchanged.
        """
        return self

    def or_else[T_local, F_local](self, func: Callable[[E], Result[T_local, F_local]]) -> Result[T_local, F_local]:
        """Call a function with the error to attempt recovery.

        Args:
            func: A function that takes the error and returns a new `Result`.

        Returns:
            The `Result` returned by `func`.

        Examples:
            >>> Err("file missing").or_else(lambda _: Ok("default content"))
            Ok('default content')
        """
        return func(self._error)

    def flatten(self) -> Err[E]:
        """Short-circuit the flattening and return self unchanged."""
        return self

    def filter(self, _predicate: Callable[[Any], bool], _error: object) -> Err[E]:
        """Ignore the filter and return self unchanged."""
        return self

    def match[U, T_local](self, on_ok: Callable[[T_local], U], on_err: Callable[[E], U]) -> U:  # noqa: ARG002
        """Exhaustively handle both success and failure cases.

        Args:
            on_ok: Ignored success handler.
            on_err: Function to call with the error state.

        Returns:
            The result of the error handler.

        Examples:
            >>> res = Err("fail")
            >>> res.match(on_ok=lambda x: x * 2, on_err=lambda e: 0)
            0
        """
        return on_err(self._error)

    def unwrap_or[T_local](self, default: T_local) -> T_local:
        """Return the provided default value.

        Args:
            default: The fallback value.

        Returns:
            The `default` value.

        Examples:
            >>> Err("fail").unwrap_or(42)
            42
        """
        return default

    def unwrap_or_else[T_local](self, func: Callable[[E], T_local]) -> T_local:
        """Call a function with the error to produce a fallback value.

        Args:
            func: A function that generates a fallback value from the error.

        Returns:
            The result of `func(error)`.

        Examples:
            >>> Err("fail").unwrap_or_else(lambda e: f"recovered from {e}")
            'recovered from fail'
        """
        return func(self._error)

    def ok(self) -> None:
        """Convert to Optional[T]. Always returns `None` for `Err` variants.

        Examples:
            >>> Err("fail").ok()
            None
        """
        return

    def err(self) -> E:
        """Convert to Optional[E]. Returns the error state.

        Returns:
            The contained error state.

        Examples:
            >>> Err("fail").err()
            'fail'
        """
        return self._error


# --- Standalone Utilities ---


def is_ok[T, E](result: Result[T, E]) -> TypeIs[Ok[T]]:
    """Check if a result is an `Ok` variant and narrow its type.

    Examples:
        >>> res: Result[int, str] = Ok(10)
        >>> if is_ok(res):
        ...     # res is now typed as Ok[int]
        ...     print(res._value)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    """
    return isinstance(result, Ok)


def is_err[T, E](result: Result[T, E]) -> TypeIs[Err[E]]:
    """Check if a result is an `Err` variant and narrow its type.

    Examples:
        >>> res: Result[int, str] = Err("fail")
        >>> if is_err(res):
        ...     # res is now typed as Err[str]
        ...     print(res._error)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    """
    return isinstance(result, Err)


def from_optional[T, E](value: T | None, error: E) -> Result[T, E]:
    """Create a Result from an optional value.

    Args:
        value: The optional value to wrap.
        error: The error to return if the value is None.

    Returns:
        Ok(value) if value is not None, otherwise Err(error).

    Examples:
        >>> from_optional(42, "fail")
        Ok(42)
        >>> from_optional(None, "fail")
        Err('fail')
    """
    if value is None:
        return Err(error)
    return Ok(value)


def combine[T, E](results: Iterable[Result[T, E]]) -> Result[list[T], E]:
    """Combine an iterable of results into a single result (All-or-Nothing).

    If all results are `Ok`, returns an `Ok` containing a list of all values.
    If any result is an `Err`, returns the first `Err` encountered.

    Args:
        results: An iterable of `Result` instances.

    Returns:
        A combined `Result`.

    Examples:
        >>> combine([Ok(1), Ok(2)])
        Ok([1, 2])
        >>> combine([Ok(1), Err("fail")])
        Err('fail')
    """
    values: list[T] = []
    for res in results:
        if isinstance(res, Err):
            return res  # pyright: ignore[reportReturnType]
        values.append(res._value)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    return Ok(values)


all_results = combine


def partition[T, E](results: Iterable[Result[T, E]]) -> tuple[list[T], list[E]]:
    """Partition an iterable of results into two lists.

    Unlike `combine`, this does not short-circuit. It collects all values
    into two separate lists.

    Args:
        results: An iterable of `Result` instances.

    Returns:
        A tuple of (ok_values, err_states).

    Examples:
        >>> partition([Ok(1), Err("a"), Ok(2)])
        ([1, 2], ['a'])
    """
    oks: list[T] = []
    errs: list[E] = []
    for res in results:
        if isinstance(res, Ok):
            oks.append(res._value)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        else:
            errs.append(res._error)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
    return oks, errs


def safe[T, **P](
    exceptions: type[Exception] | tuple[type[Exception], ...],
    func: Callable[P, T] | None = None,
) -> Any:  # noqa: ANN401
    """Execute a function and catch specified exceptions into a Result.

    Can be used as a standalone wrapper or as a decorator.

    Args:
        exceptions: An exception type or tuple of types to catch.
        func: Optional function to wrap and execute immediately.

    Returns:
        A Result if `func` was provided, otherwise a decorator.

    Examples:
        >>> @safe(ValueError)
        ... def parse(s: str) -> int:
        ...     return int(s)
        >>> parse("10")
        Ok(10)
        >>> parse("not an int")
        Err(ValueError(...))

        >>> safe(ValueError, int, "10")
        Ok(10)
    """

    def decorator(f: Callable[P, T]) -> Callable[P, Any]:
        if inspect.iscoroutinefunction(f):

            @wraps(f)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[Any, Exception]:
                try:
                    return Ok(await f(*args, **kwargs))
                except exceptions as e:
                    # Ensure it is a standard Exception for the Result[T, Exception] type
                    assert isinstance(e, Exception)
                    res: Result[Any, Exception] = Err(e)
                    return res

            return async_wrapper

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, Exception]:
            try:
                return Ok(f(*args, **kwargs))
            except exceptions as e:
                # Ensure it is a standard Exception for the Result[T, Exception] type
                assert isinstance(e, Exception)
                res: Result[T, Exception] = Err(e)
                return res

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def do[T, E](gen: Generator[Result[T, E], Any, T]) -> Result[T, E]:
    """Helper for inline synchronous do-notation (generator expressions).

    If an 'Err' is encountered, it short-circuits and returns that error.

    Args:
        gen: A generator expression yielding Results.

    Returns:
        The final Result of the chain.

    Examples:
        >>> do(Ok(x + y) for x in Ok(1) for y in Ok(2))
        Ok(3)
    """
    try:
        return next(gen)
    except _DoError as e:
        return e.err
    except StopIteration as e:
        # If the generator finishes normally, wrap its return value in Ok
        return Ok(e.value)


async def do_async[T, E](gen: AsyncGenerator[Result[T, E], Any]) -> Result[T, E]:
    """Helper for inline asynchronous do-notation (generator expressions).

    If an 'Err' is encountered, it short-circuits and returns that error.

    Args:
        gen: An async generator expression yielding Results.

    Returns:
        The final Result of the chain.

    Examples:
        >>> async def async_gen():
        ...     yield Ok(1)
        >>> await do_async(Ok(x + 2) async for x in async_gen())
        Ok(3)
    """
    try:
        return await anext(gen)
    except _DoError as e:
        return e.err
    except StopAsyncIteration:
        msg = "Async do-notation generator ended without yielding a Result"
        raise UnwrapError(Ok(None), msg) from None


# --- do-notation style decorators ---


def _make_do_wrapper[T, E, **P](
    func: Callable[P, Do[T, E]],
    catch_types: type[Exception] | tuple[type[Exception], ...] | None,
) -> Callable[P, Result[T, E | Exception]]:
    """Internal helper to drive the generator-based bind simulation."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, E | Exception]:
        try:
            gen = func(*args, **kwargs)
            res = next(gen)
            while True:
                if isinstance(res, Err):
                    return res  # pyright: ignore[reportReturnType]
                res = gen.send(res._value)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
        except StopIteration as e:
            return Ok(e.value)  # e.value is standard for StopIteration
        except Exception as e:
            if catch_types and isinstance(e, catch_types):
                return Err(e)
            raise

    return wrapper


def do_notation[T, E, **P](
    arg: Callable[P, Do[T, E]] | type[Exception] | tuple[type[Exception], ...] | None = None,
    *,
    catch: type[Exception] | tuple[type[Exception], ...] | None = None,
) -> Any:  # noqa: ANN401
    """Enable imperative-style 'do-notation' for synchronous Result blocks.

    This decorator allows writing code that looks procedural by using `yield`
    to unwrap `Result` values. If any `yield` receives an `Err`, the function
    short-circuits immediately and returns that Err.

    Notes:
        The final 'return' value is automatically wrapped in an Ok variant.

    Args:
        arg: The function to decorate, or an exception type to catch.
        catch: Optional keyword-only exception type to catch and lift into Err.

    Returns:
        The decorated function or a decorator factory.

    Examples:
        >>> @do_notation
        ... def process(x):
        ...     a = yield Ok(x + 1)
        ...     b = yield Ok(a * 2)
        ...     return b
        >>> process(5)
        Ok(12)

        >>> @do_notation(catch=ValueError)
        ... def risky(s):
        ...     val = yield Ok(int(s))
        ...     return val
        >>> risky("not a number")
        Err(ValueError(...))
    """
    if callable(arg) and not isinstance(arg, type | tuple):
        return _make_do_wrapper(arg, None)
    catch_final = catch or (arg if isinstance(arg, type | tuple) else None)  # pyright: ignore[reportUnknownVariableType]

    def decorator(func: Callable[P, Do[T, E]]) -> Callable[P, Result[T, E | Exception]]:
        return _make_do_wrapper(func, catch_final)  # type: ignore[arg-type] # pyright: ignore[reportReturnType]

    return decorator


def _make_async_wrapper[T, E, **P](
    func: Callable[P, DoAsync[T, E]],
    catch_types: type[Exception] | tuple[type[Exception], ...] | None,
) -> Callable[P, Coroutine[Any, Any, Result[T, E | Exception]]]:
    """Internal helper to drive the async generator-based bind simulation."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, E | Exception]:
        try:
            gen = func(*args, **kwargs)
            last_val: Any = None
            try:
                res = await anext(gen)
                while True:
                    last_val = res
                    if isinstance(res, Err):
                        await gen.aclose()
                        return res  # pyright: ignore[reportReturnType]
                    res = await gen.asend(res._value)  # pyright: ignore[reportPrivateUsage] # noqa: SLF001
            except StopAsyncIteration:
                if last_val is None:
                    msg = "Async do-notation must yield at least one value"
                    raise RuntimeError(msg) from None
                return last_val
        except Exception as e:
            if catch_types and isinstance(e, catch_types):
                return Err(e)
            raise

    return wrapper


def do_notation_async[T, E, **P](
    arg: Callable[P, DoAsync[T, E]] | type[Exception] | tuple[type[Exception], ...] | None = None,
    *,
    catch: type[Exception] | tuple[type[Exception], ...] | None = None,
) -> Any:  # noqa: ANN401
    """Enable imperative-style 'do-notation' for asynchronous Result blocks.

    This is the async version of `@do_notation`. It supports `yield await` for
    asynchronous Result-returning functions.

    Notes:
        Unlike the sync version, the final result MUST be yielded as an Ok
        variant because async generators do not support native return values
        in this context.

    Args:
        arg: The async function to decorate, or an exception type to catch.
        catch: Optional keyword-only exception type to catch and lift into Err.

    Returns:
        The decorated async function or a decorator factory.

    Examples:
        >>> @do_notation_async
        ... async def get_data(user_id):
        ...     user = yield await fetch_user(user_id)  # fetch_user returns Result
        ...     yield Ok(user.name)
    """

    def decorator(func: Callable[P, DoAsync[T, E]]) -> Any:  # noqa: ANN401
        return _make_async_wrapper(func, catch_final)  # type: ignore[arg-type]

    if callable(arg) and not isinstance(arg, type | tuple):
        return _make_async_wrapper(arg, None)

    catch_final = catch or (arg if isinstance(arg, type | tuple) else None)  # pyright: ignore[reportUnknownVariableType]
    return decorator


# --- Internal Teaching Helper ---


def _raise_api_error(method_name: str) -> NoReturn:
    """Raise a descriptive error to guide users to the correct API."""
    match method_name:
        case "value" | "error":
            msg = (
                f"Result API Warning: Direct access to '.{method_name}' is disabled to prevent unhandled errors. "
                "Use pattern matching ('match res: case Ok(v): ...'), functional methods like '.map()', "
                "or safe conversion methods like '.ok()' and '.err()'."
            )
        case "inspect" | "inspect_async" | "inspect_err":
            target = method_name.replace("inspect", "tap")
            msg = (
                f"Result API Warning: '.{method_name}()' has been renamed to '.{target}()' to align with "
                "modern functional naming conventions. Please update your code to use the new naming."
            )
        case "unwrap" | "unwrap_err" | "expect" | "expect_err" | "unwrap_or_raise":
            msg = (
                f"Result API Warning: '.{method_name}()' is a crashing operation and is isolated "
                "in the '.unsafe' namespace to encourage safe functional patterns. "
                f"Use '.unsafe.{method_name}()' if you specifically need to panic, or prefer "
                "safe alternatives like '.unwrap_or()' or '.unwrap_or_else()'."
            )
        case _:
            msg = f"Result API Warning: '.{method_name}()' is not part of the supported Result API."

    raise AttributeError(msg)

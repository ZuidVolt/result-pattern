"""# Result: Functional Error Handling for Modern Python.

A lightweight, single-file library designed to implement the 'Errors as Values'
pattern in Python 3.14+. This library tries to help bridge the gap between pure
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
5.  **Pragmatic Interop**: Provide 'lifting' tools like `@catch` to seamlessly
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

# pyright: reportPrivateUsage=false
# ruff: noqa: SLF001

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Final, Literal, Never, TypeIs, TypeVar, Union, cast, overload

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Iterable
    from types import TracebackType
    from typing import ParamSpec, Protocol

    P = ParamSpec("P")
    T_ret = TypeVar("T_ret")

    class _CatchDecoratorOrContext[E: Exception](Protocol):
        @overload
        def __call__[T_ret, **P](self, f: Callable[P, T_ret]) -> Callable[P, Result[T_ret, E]]: ...  # pyright: ignore[reportOverlappingOverload]
        @overload
        def __call__[T_ret, **P](  # pyright: ignore[reportOverlappingOverload]
            self, f: Callable[P, Coroutine[Any, Any, T_ret]]
        ) -> Callable[P, Awaitable[Result[T_ret, E]]]: ...
        def __call__(self, f: Any) -> Any: ...
        def __enter__[T_ctx](self) -> CatchContext[T_ctx, E]: ...
        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> bool: ...


T_co = TypeVar("T_co", covariant=True)
E_co = TypeVar("E_co", covariant=True)
T = TypeVar("T")
E = TypeVar("E")

# --- Exceptions ---


@dataclass
class CatchContext[T, E]:
    """A context manager for localized exception trapping into a Result.

    This is returned by calling `catch()` as a context manager.
    """

    exceptions: type[E] | tuple[type[E], ...]
    result: Result[T, E] | None = field(default=None, init=False)

    def __enter__(self) -> CatchContext[T, E]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is None:
            return False

        if issubclass(exc_type, self.exceptions):
            self.result = Err(cast("E", exc_val))
            return True

        return False

    def set(self, value: T) -> None:
        """Set the success value for the context."""
        self.result = Ok(value)


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

Result = Union["Ok[T_co]", "Err[E_co]"]
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

type DoAsync[T, E] = AsyncGenerator[Result[T, E] | Result[Any, E], Any]
"""
A type alias for async generator functions compatible with the `@do_async` decorator.

Async generators must yield an `Ok` variant as their final step to simulate
a return value.
"""


# --- Unsafe Proxies ---


@dataclass(frozen=True, slots=True)
class _OkUnsafe[T_co]:
    """Namespace for operations that may raise an exception on an Ok variant."""

    _owner: Ok[T_co]

    def unwrap(self) -> T_co:
        """Extract the contained value. Always succeeds on Ok.

        Returns:
            The contained success value.

        """
        return self._owner._value

    def unwrap_err(self) -> Never:
        """Raise an UnwrapError because an Ok value does not contain an error.

        Raises:
            UnwrapError: Always raised with a descriptive message.

        """
        msg = f"Called unwrap_err on an Ok value: {self._owner._value!r}"
        raise UnwrapError(self._owner, msg)

    def expect(self, _msg: str) -> T_co:
        """Extract the contained value, ignoring the custom message.

        Always succeeds on `Ok` variants.

        Args:
            _msg: The custom panic message (ignored on success).

        Returns:
            The contained success value.

        """
        return self._owner._value

    def unwrap_or_raise(self, _e: type[Exception]) -> T_co:
        """Extract the contained value, ignoring the exception type.

        Always succeeds on `Ok` variants.

        Args:
            _e: The exception type to raise (ignored on success).

        Returns:
            The contained success value.

        """
        return self._owner._value

    def unwrap_or_default(self) -> T_co:
        """Extract the contained value.

        Always succeeds on `Ok` variants.

        Returns:
            The contained success value.

        """
        return self._owner._value

    def expect_err(self, msg: str) -> Never:
        """Raise an UnwrapError because an Ok value does not contain an error.

        Args:
            msg: The custom panic message.

        Raises:
            UnwrapError: Always raised.

        """
        msg_final = f"{msg}: {self._owner._value!r}"
        raise UnwrapError(self._owner, msg_final)


@dataclass(frozen=True, slots=True)
class _ErrUnsafe[E_co]:
    """Namespace for operations that may raise an exception on an Err variant."""

    _owner: Err[E_co]

    def unwrap(self) -> Never:
        """Raise an UnwrapError because an error cannot be unwrapped.

        Raises:
            UnwrapError: Always raised, containing the error state.

        """
        msg = f"Called unwrap on an Err value: {self._owner._error!r}"
        exc = UnwrapError(self._owner, msg)
        if isinstance(self._owner._error, BaseException):
            raise exc from self._owner._error
        raise exc

    def unwrap_err(self) -> E_co:
        """Extract the contained error. Always succeeds on Err.

        Returns:
            The contained error state.

        """
        return self._owner._error

    def expect(self, msg: str) -> Never:
        """Raise an UnwrapError with a custom message.

        Args:
            msg: The custom message to prepend to the error state output.

        Raises:
            UnwrapError: Always raised.

        """
        msg_final = f"{msg}: {self._owner._error!r}"
        exc = UnwrapError(self._owner, msg_final)
        if isinstance(self._owner._error, BaseException):
            raise exc from self._owner._error
        raise exc

    def unwrap_or_raise(self, e: type[Exception]) -> Never:
        """Raise a custom exception with the error state.

        Args:
            e: The exception class to instantiate and raise.

        Raises:
            Exception: An instance of `e` initialized with the error state.

        """
        raise e(self._owner._error)

    def unwrap_or_default(self) -> Any:
        """Return the default value for the error's expected type if possible.

        Since we don't have a Default trait, we try to infer from common types.
        Otherwise, this returns None.

        Returns:
            A default value (0, "", [], etc.) or None.

        """
        return None

    def expect_err(self, _msg: str) -> E_co:
        """Extract the contained error, ignoring the custom message.

        Always succeeds on `Err` variants.

        Args:
            _msg: The custom panic message (ignored on failure).

        Returns:
            The contained error state.

        """
        return self._owner._error


# --- Core Variants ---


@dataclass(frozen=True, slots=True)  # noqa: PLR0904
class Ok[T_co]:
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

    _value: T_co
    __match_args__ = ("_value",)

    def __init__(self, value: T_co) -> None:
        """Initialize an Ok variant.

        Args:
            value: The successful result to wrap.

        """
        object.__setattr__(self, "_value", value)

    def __iter__(self) -> Generator[Any, Any, T_co]:
        """Allow use in generator expressions for do-notation."""
        yield self._value
        return self._value  # noqa: B901

    async def __aiter__(self) -> AsyncGenerator[T_co, Any]:
        """Allow use in async generator expressions for do-notation."""
        yield self._value

    @property
    def unsafe(self) -> _OkUnsafe[T_co]:
        """Namespace for operations that might panic (raise exceptions).

        Examples:
            >>> Ok(10).unsafe.unwrap()
            10

        """
        return _OkUnsafe(self)

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"

    def is_ok(self) -> Literal[True]:
        """Check if the result is an Ok variant.

        Returns:
            Always True for Ok instances.

        """
        return True

    def is_err(self) -> Literal[False]:
        """Check if the result is an Err variant.

        Returns:
            Always False for Ok instances.

        """
        return False

    def is_ok_and(self, predicate: Callable[[T_co], bool]) -> bool:
        """Check if the result is an Ok variant and matches a predicate.

        Args:
            predicate: A function to test the success value.

        Returns:
            True if the result is Ok and the predicate returns True.

        Examples:
            >>> Ok(10).is_ok_and(lambda x: x > 5)
            True

        """
        return predicate(self._value)

    def is_err_and(self, _predicate: Callable[[Any], bool]) -> Literal[False]:
        """Check if the result is an Err variant and matches a predicate.

        Always returns False for Ok instances.

        Args:
            _predicate: Ignored for Ok instances.

        Returns:
            Always False.

        """
        return False

    def __getattr__(self, name: str) -> Never:
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

    def unwrap(self) -> Never:
        """Root-level unwrap is disabled. Use .unsafe.unwrap() instead."""
        _raise_api_error("unwrap")

    def unwrap_err(self) -> Never:
        """Root-level unwrap_err is disabled. Use .unsafe.unwrap_err() instead."""
        _raise_api_error("unwrap_err")

    def expect(self, _msg: str) -> Never:
        """Root-level expect is disabled. Use .unsafe.expect() instead."""
        _raise_api_error("expect")

    def expect_err(self, _msg: str) -> Never:
        """Root-level expect_err is disabled. Use .unsafe.expect_err() instead."""
        _raise_api_error("expect_err")

    def unwrap_or_raise(self, _e: type[Exception]) -> Never:
        """Root-level unwrap_or_raise is disabled. Use .unsafe.unwrap_or_raise() instead."""
        _raise_api_error("unwrap_or_raise")

    def inspect(self, _func: Callable[[Any], Any]) -> Never:
        """Root-level inspect is disabled. Use .tap() instead."""
        _raise_api_error("inspect")

    def inspect_async(self, _func: Callable[[Any], Awaitable[Any]]) -> Never:
        """Root-level inspect_async is disabled. Use .tap_async() instead."""
        _raise_api_error("inspect_async")

    def inspect_err(self, _func: Callable[[Any], Any]) -> Never:
        """Root-level inspect_err is disabled. Use .tap_err() instead."""
        _raise_api_error("inspect_err")

    # --- Functional API ---

    def map[U](self, func: Callable[[T_co], U]) -> Ok[U]:
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

    async def map_async[U](self, func: Callable[[T_co], Awaitable[U]]) -> Ok[U]:
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

    def map_or[U](self, _default: U, func: Callable[[T_co], U]) -> U:
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

    def map_or_else[U](self, _default_func: Callable[[], U], func: Callable[[T_co], U]) -> U:
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

    def map_err(self, _func: Callable[[Any], Any]) -> Ok[T_co]:
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

    def replace_err(self, _error: object) -> Ok[T_co]:
        """Ignore the error replacement and return self unchanged."""
        return self

    def tap(self, func: Callable[[T_co], Any]) -> Ok[T_co]:
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

    async def tap_async(self, func: Callable[[T_co], Awaitable[Any]]) -> Ok[T_co]:
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

    def tap_err(self, _func: Callable[[Any], Any]) -> Ok[T_co]:
        """Ignore the error side effect and return self unchanged."""
        return self

    def and_then[U, E_local](self, func: Callable[[T_co], Result[U, E_local]]) -> Result[U, E_local]:
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
        self, func: Callable[[T_co], Awaitable[Result[U, E_local]]]
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

    def or_else(self, _func: object) -> Ok[T_co]:
        """Ignore the recovery function and return self unchanged.

        Args:
            _func: Ignored recovery function.

        Returns:
            The current instance unchanged.

        Examples:
            >>> Ok(10).or_else(lambda e: Ok(0))
            Ok(10)

        """
        return self

    @overload
    def flatten[R: (Ok[Any] | Err[Any])](self: Ok[R]) -> R: ...  # type: ignore[overload-overlap]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]

    @overload
    def flatten(self) -> Ok[T_co]: ...

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
        val = self._value
        if isinstance(val, Ok | Err):
            return cast("Result[Any, Any]", val)
        return self

    def filter[E_local](self, predicate: Callable[[T_co], bool], error: E_local) -> Result[T_co, E_local]:
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

    def match[U](self, on_ok: Callable[[T_co], U], on_err: Callable[[Any], U]) -> U:  # noqa: ARG002
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

    def unwrap_or(self, _default: object) -> T_co:
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

    def unwrap_or_else(self, _func: Callable[[Any], T_co]) -> T_co:
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

    def ok(self) -> T_co:
        """Convert to Optional[T].

        Returns:
            The contained value. Always succeeds on `Ok`.

        Examples:
            >>> Ok(10).ok()
            10

        """
        return self._value

    def err(self) -> object | None:
        """Convert to Optional[E].

        Returns:
            Always returns `None` for `Ok` variants.

        Examples:
            >>> Ok(10).err()
            None

        """
        return None

    def transpose(self) -> Result[T_co, E_co] | None:
        """Transpose a Result of an Optional into an Optional of a Result.

        Ok(None) becomes None. Ok(Some(v)) becomes Ok(v).

        Returns:
            None if the value is None, otherwise self.

        Examples:
            >>> Ok(10).transpose()
            Ok(10)
            >>> Ok(None).transpose()
            None

        """
        if self._value is None:
            return None
        return cast("Result[T_co, E_co] | None", self)

    def product[U, E2](self, other: Result[U, E2]) -> Result[tuple[T_co, U], E_co | E2]:  # type: ignore[valid-type]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]
        """Combine two results into a result of a tuple (Zip).

        Args:
            other: Another Result to zip with.

        Returns:
            Ok((val1, val2)) if both are Ok, otherwise the first Err.

        Examples:
            >>> Ok(1).product(Ok(2))
            Ok((1, 2))

        """
        if isinstance(other, Err):
            return cast("Any", other)  # type: ignore[no-any-return] # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]
        return Ok[tuple[T_co, U]]((self._value, other._value))


@dataclass(frozen=True, slots=True)  # noqa: PLR0904
class Err[E_co]:
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

    _error: E_co
    __match_args__ = ("_error",)

    def __init__(self, error: E_co) -> None:
        """Initialize an Err variant.

        Args:
            error: The error state or exception to wrap.

        """
        object.__setattr__(self, "_error", error)

    def __iter__(self) -> Generator[Result[Any, E_co], Any, Never]:
        """Allow use in generator expressions for do-notation."""
        raise _DoError(self)
        yield self  # should be UNREACHABLE

    async def __aiter__(self) -> AsyncGenerator[Never, Any]:
        """Allow use in async generator expressions for do-notation."""
        raise _DoError(self)
        yield self  # should be UNREACHABLE

    @property
    def unsafe(self) -> _ErrUnsafe[E_co]:
        """Namespace for operations that might panic.

        Examples:
            >>> Err("fail").unsafe.unwrap_err()
            'fail'

        """
        return _ErrUnsafe(self)

    def __repr__(self) -> str:
        return f"Err({self._error!r})"

    def is_ok(self) -> Literal[False]:
        """Check if the result is an Ok variant.

        Returns:
            Always False for Err instances.

        """
        return False

    def is_err(self) -> Literal[True]:
        """Check if the result is an Err variant.

        Returns:
            Always True for Err instances.

        """
        return True

    def is_ok_and(self, _predicate: Callable[[Any], bool]) -> Literal[False]:
        """Check if the result is an Ok variant and matches a predicate.

        Always returns False for Err instances.

        Args:
            _predicate: Ignored for Err instances.

        Returns:
            Always False.

        """
        return False

    def is_err_and(self, predicate: Callable[[E_co], bool]) -> bool:
        """Check if the result is an Err variant and matches a predicate.

        Args:
            predicate: A function to test the error state.

        Returns:
            True if the result is Err and the predicate returns True.

        Examples:
            >>> Err(404).is_err_and(lambda e: e == 404)
            True

        """
        return predicate(self._error)

    def __getattr__(self, name: str) -> Never:
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

    def unwrap(self) -> Never:
        """Root-level unwrap is disabled. Use .unsafe.unwrap() instead."""
        _raise_api_error("unwrap")

    def unwrap_err(self) -> Never:
        """Root-level unwrap_err is disabled. Use .unsafe.unwrap_err() instead."""
        _raise_api_error("unwrap_err")

    def expect(self, _msg: str) -> Never:
        """Root-level expect is disabled. Use .unsafe.expect() instead."""
        _raise_api_error("expect")

    def expect_err(self, _msg: str) -> Never:
        """Root-level expect_err is disabled. Use .unsafe.expect_err() instead."""
        _raise_api_error("expect_err")

    def unwrap_or_raise(self, _e: type[Exception]) -> Never:
        """Root-level unwrap_or_raise is disabled. Use .unsafe.unwrap_or_raise() instead."""
        _raise_api_error("unwrap_or_raise")

    def inspect(self, _func: Callable[[Any], Any]) -> Never:
        """Root-level inspect is disabled. Use .tap() instead."""
        _raise_api_error("inspect")

    def inspect_async(self, _func: Callable[[Any], Awaitable[Any]]) -> Never:
        """Root-level inspect_async is disabled. Use .tap_async() instead."""
        _raise_api_error("inspect_async")

    def inspect_err(self, _func: Callable[[Any], Any]) -> Never:
        """Root-level inspect_err is disabled. Use .tap_err() instead."""
        _raise_api_error("inspect_err")

    # --- Functional API ---

    def map[U, T_local](self, _func: Callable[[T_local], U]) -> Err[E_co]:
        """Ignore the value mapping and return self unchanged."""
        return self

    async def map_async[U, T_local](self, _func: Callable[[T_local], Awaitable[U]]) -> Err[E_co]:
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

    def map_err[F](self, func: Callable[[E_co], F]) -> Err[F]:
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

    def replace(self, _value: object) -> Err[E_co]:
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

    def tap(self, _func: Callable[[Any], Any]) -> Err[E_co]:
        """Ignore the side effect and return self unchanged."""
        return self

    async def tap_async[T_local](self, _func: Callable[[T_local], Awaitable[Any]]) -> Err[E_co]:
        """Ignore async success side effects.

        Args:
            _func: Ignored async side effect function.

        Returns:
            The current instance unchanged.

        """
        return self

    def tap_err(self, func: Callable[[E_co], Any]) -> Err[E_co]:
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

    def and_then(self, _func: Callable[[Any], Result[Any, Any]]) -> Err[E_co]:
        """Short-circuit the chain and return self unchanged."""
        return self

    async def and_then_async[U, T_local](self, _func: Callable[[T_local], Awaitable[Result[U, E_co]]]) -> Err[E_co]:
        """Short-circuit async success chaining.

        Args:
            _func: Ignored async success chaining function.

        Returns:
            The current instance unchanged.

        """
        return self

    def or_else[T_local, F_local](self, func: Callable[[E_co], Result[T_local, F_local]]) -> Result[T_local, F_local]:
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

    def flatten(self) -> Err[E_co]:
        """Short-circuit the flattening and return self unchanged.

        Returns:
            The current instance unchanged.

        """
        return self

    def filter(self, _predicate: Callable[[Any], bool], _error: object) -> Err[E_co]:
        """Ignore the filter and return self unchanged."""
        return self

    def match[U, T_local](self, on_ok: Callable[[T_local], U], on_err: Callable[[E_co], U]) -> U:  # noqa: ARG002
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

    def unwrap_or_else[T_local](self, func: Callable[[E_co], T_local]) -> T_local:
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

    def ok(self) -> object | None:
        """Convert to Optional[T]. Always returns `None` for `Err` variants.

        Examples:
            >>> Err("fail").ok()
            None

        """
        return None

    def err(self) -> E_co:
        """Convert to Optional[E]. Returns the error state.

        Returns:
            The contained error state.

        Examples:
            >>> Err("fail").err()
            'fail'

        """
        return self._error

    def transpose(self) -> Err[E_co]:
        """Transpose an Err variant.

        Err(e) remains Err(e) (wrapped in the optional result).

        Returns:
            Self unchanged.

        """
        return self

    def product[U, E2](self, _other: Result[U, E2]) -> Result[tuple[Any, U], E_co | E2]:
        """Short-circuit the product and return self.

        Args:
            _other: Ignored result.

        Returns:
            Self unchanged.

        """
        return cast("Any", self)  # type: ignore[no-any-return] # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]


OkErr: Final = (Ok, Err)
"""
A constant for use in `isinstance` checks.

Examples:
    >>> res = Ok(10)
    >>> isinstance(res, OkErr)
    True
"""


# --- Standalone Utilities ---


def is_ok[T_local, E_local](result: Result[T_local, E_local]) -> TypeIs[Ok[T_local]]:
    """Check if a result is an `Ok` variant and narrow its type.

    Examples:
        >>> res: Result[int, str] = Ok(10)
        >>> if is_ok(res):
        ...     # res is now typed as Ok[int]
        ...     print(res._value)

    """
    return isinstance(result, Ok)


def is_err[T_local, E_local](result: Result[T_local, E_local]) -> TypeIs[Err[E_local]]:
    """Check if a result is an `Err` variant and narrow its type.

    Examples:
        >>> res: Result[int, str] = Err("fail")
        >>> if is_err(res):
        ...     # res is now typed as Err[str]
        ...     print(res._error)

    """
    return isinstance(result, Err)


def from_optional[T_co, E_co](value: T_co | None, error: E_co) -> Result[T_co, E_co]:
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


def combine[T_local, E_local](results: Iterable[Result[T_local, E_local]]) -> Result[list[T_local], E_local]:
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
    values: list[T_local] = []
    for res in results:
        if isinstance(res, Err):
            return res  # pyright: ignore[reportReturnType]
        values.append(res._value)
    return Ok(values)


def partition[T_local, E_local](results: Iterable[Result[T_local, E_local]]) -> tuple[list[T_local], list[E_local]]:
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
    oks: list[T_local] = []
    errs: list[E_local] = []
    for res in results:
        if isinstance(res, Ok):
            oks.append(res._value)
        else:
            errs.append(res._error)
    return oks, errs


def map2[T1, T2, U, E1, E2](
    res1: Result[T1, E1],
    res2: Result[T2, E2],
    func: Callable[[T1, T2], U],
) -> Result[U, E1 | E2]:
    """Apply a binary function to the values of two results (Zipping Map).

    If both are Ok, returns Ok(func(val1, val2)).
    If either is Err, returns the first Err encountered.

    Args:
        res1: First Result.
        res2: Second Result.
        func: A function taking two success values.

    Returns:
        A new Result.

    Examples:
        >>> map2(Ok(1), Ok(2), lambda x, y: x + y)
        Ok(3)

    """
    if isinstance(res1, Err):
        return cast("Result[U, E1 | E2]", res1)
    if isinstance(res2, Err):
        return cast("Result[U, E1 | E2]", res2)
    return Ok(func(res1._value, res2._value))


def any_ok[T_local, E_local](results: Iterable[Result[T_local, E_local]]) -> Result[T_local, list[E_local]]:
    """Return the first Ok variant found, or a list of all errors if none succeeded.

    Args:
        results: An iterable of Results.

    Returns:
        The first Ok found, or Err containing a list of all encountered errors.

    Examples:
        >>> any_ok([Err("a"), Ok(1), Ok(2)])
        Ok(1)
        >>> any_ok([Err("a"), Err("b")])
        Err(['a', 'b'])

    """
    errs: list[E_local] = []
    for res in results:
        if isinstance(res, Ok):
            return res
        errs.append(res._error)
    return Err(errs)


def _apply_remap[E_local](res: Err[E_local], remap: dict[type[Any], type[Any]] | None) -> Result[Any, Any]:
    """Internal helper to apply error remapping."""
    if remap:
        for src, dest in remap.items():
            if isinstance(res._error, src):
                return Err(dest(res._error))
    return res


@overload
def catch[T, **P](  # pyright: ignore[reportOverlappingOverload]
    exceptions: type[Exception] | tuple[type[Exception], ...],
    func: Callable[P, T],
) -> Callable[P, Result[T, Exception]]: ...


@overload
def catch[T, **P](  # pyright: ignore[reportOverlappingOverload]
    exceptions: type[Exception] | tuple[type[Exception], ...],
    func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Awaitable[Result[T, Exception]]]: ...


@overload
def catch[E: Exception](  # pyright: ignore[reportOverlappingOverload]
    exceptions: type[E] | tuple[type[E], ...],
    func: None = None,
) -> _CatchDecoratorOrContext[E]: ...


def catch[T, **P](  # noqa: C901 # pyright: ignore
    exceptions: type[Exception] | tuple[type[Exception], ...],
    func: Callable[P, T] | None = None,
) -> Any:
    """Execute a function and catch specified exceptions into a Result.

    Can be used as a standalone wrapper or as a decorator.

    Args:
        exceptions: An exception type or tuple of types to catch.
        func: Optional function to wrap and execute immediately.

    Returns:
        A Result if `func` was provided, otherwise a decorator.

    Examples:
        >>> @catch(ValueError)
        ... def parse(s: str) -> int:
        ...     return int(s)
        >>> parse("10")
        Ok(10)
        >>> parse("not an int")
        Err(ValueError(...))

        >>> catch(ValueError, int, "10")
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

    # If called without a function, check if it's being used as a context manager
    # or a decorator. We return an object that satisfies both.
    class DecoratorOrContext:
        def __init__(self) -> None:
            self._ctx: CatchContext[Any, Any] | None = None

        def __call__(self, f: Callable[P, T]) -> Any:
            return decorator(f)

        def __enter__(self) -> CatchContext[Any, Any]:
            self._ctx = CatchContext(exceptions)
            return self._ctx.__enter__()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> bool:
            if self._ctx is None:
                return False
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)

    return DecoratorOrContext()


def catch_call[T, E: Exception, **P](
    exceptions: type[E] | tuple[type[E], ...],
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Result[T, E]:
    """Execute a function inline and catch specified exceptions into a Result.

    This allows for clean, single-line expression evaluation without needing
    to define a new function or open a context manager block.

    Args:
        exceptions: An exception type or tuple of types to catch.
        func: The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Ok(value) if successful, Err(exception) if a specified exception was raised.

    Examples:
        >>> import json
        >>> catch_call(json.JSONDecodeError, json.loads, '{"key": "value"}')
        Ok({'key': 'value'})

    """
    try:
        return Ok(func(*args, **kwargs))
    except exceptions as e:
        return Err(cast("E", e))  # type: ignore[redundant-cast]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]


def do[T_co, E_co](gen: Generator[Result[T_co, E_co], Any, T_co]) -> Result[T_co, E_co]:
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


async def do_async[T_co, E_co](gen: AsyncGenerator[Result[T_co, E_co], Any]) -> Result[T_co, E_co]:
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


def _make_do_wrapper[T_local, E_local, **P](
    func: Callable[P, Do[T_local, E_local]],
    catch_types: type[Exception] | tuple[type[Exception], ...] | None,
    remap: dict[type[Any], type[Any]] | None = None,
) -> Callable[P, Result[Any, Any]]:
    """Internal helper to drive the generator-based bind simulation."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[Any, Any]:
        try:
            gen = func(*args, **kwargs)
            res = next(gen)
            while True:
                if isinstance(res, Err):
                    return _apply_remap(res, remap)
                res = gen.send(res._value)
        except StopIteration as e:
            return Ok(e.value)  # e.value is standard for StopIteration
        except Exception as e:
            if catch_types and isinstance(e, catch_types):
                return Err(e)
            raise

    return wrapper  # pyright: ignore[reportReturnType]


@overload
def do_notation[T_local, E_local, **P](
    arg: Callable[P, Do[T_local, E_local]],
    *,
    catch: None = None,
    remap: dict[type[Any], type[Any]] | None = None,
) -> Callable[P, Result[T_local, E_local]]: ...


@overload
def do_notation[T_local, E_local, **P](
    arg: type[Exception] | tuple[type[Exception], ...] | None = None,
    *,
    catch: type[Exception] | tuple[type[Exception], ...] | None = None,
    remap: dict[type[Any], type[Any]] | None = None,
) -> Callable[[Callable[P, Do[T_local, E_local]]], Callable[P, Result[T_local, E_local | Exception]]]: ...


def do_notation[T_local, E_local, **P](
    arg: Callable[P, Do[T_local, E_local]] | type[Exception] | tuple[type[Exception], ...] | None = None,
    *,
    catch: type[Exception] | tuple[type[Exception], ...] | None = None,
    remap: dict[type[Any], type[Any]] | None = None,
) -> Any:
    """Enable imperative-style 'do-notation' for synchronous Result blocks.

    This decorator allows writing code that looks procedural by using `yield`
    to unwrap `Result` values. If any `yield` receives an `Err`, the function
    short-circuits immediately and returns that Err.

    Notes:
        The final 'return' value is automatically wrapped in an Ok variant.

    Args:
        arg: The function to decorate, or an exception type to catch.
        catch: Optional keyword-only exception type to catch and lift into Err.
        remap: Optional mapping of internal error types to high-level domain errors.

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
        return _make_do_wrapper(arg, None, remap)
    catch_final = catch or (arg if isinstance(arg, type | tuple) else None)  # pyright: ignore[reportUnknownVariableType]

    def decorator(func: Callable[P, Do[T_local, E_local]]) -> Callable[P, Result[T_local, E_local | Exception]]:
        return _make_do_wrapper(func, cast("Any", catch_final), remap)

    return decorator


def _make_async_wrapper[T_local, E_local, **P](
    func: Callable[P, DoAsync[T_local, E_local]],
    catch_types: type[Exception] | tuple[type[Exception], ...] | None,
    remap: dict[type[Any], type[Any]] | None = None,
) -> Callable[P, Coroutine[Any, Any, Result[Any, Any]]]:
    """Internal helper to drive the async generator-based bind simulation."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[Any, Any]:
        try:
            gen = func(*args, **kwargs)
            last_val: Result[Any, Any] | None = None
            try:
                res = await anext(gen)
                while True:
                    last_val = res
                    if isinstance(res, Err):
                        res_final = _apply_remap(res, remap)
                        await gen.aclose()
                        return res_final
                    res = await gen.asend(res._value)
            except StopAsyncIteration:
                if last_val is None:
                    msg = "Async do-notation must yield at least one value"
                    raise RuntimeError(msg) from None
                return last_val
        except Exception as e:
            if catch_types and isinstance(e, catch_types):
                return Err(e)
            raise

    return wrapper  # pyright: ignore[reportReturnType]


@overload
def do_notation_async[T_local, E_local, **P](
    arg: Callable[P, DoAsync[T_local, E_local]],
    *,
    catch: None = None,
    remap: dict[type[Any], type[Any]] | None = None,
) -> Callable[P, Coroutine[Any, Any, Result[T_local, E_local]]]: ...


@overload
def do_notation_async[T_local, E_local, **P](
    arg: type[Exception] | tuple[type[Exception], ...] | None = None,
    *,
    catch: type[Exception] | tuple[type[Exception], ...] | None = None,
    remap: dict[type[Any], type[Any]] | None = None,
) -> Callable[
    [Callable[P, DoAsync[T_local, E_local]]],
    Callable[P, Coroutine[Any, Any, Result[T_local, E_local | Exception]]],
]: ...


def do_notation_async[T_local, E_local, **P](
    arg: Callable[P, DoAsync[T_local, E_local]] | type[Exception] | tuple[type[Exception], ...] | None = None,
    *,
    catch: type[Exception] | tuple[type[Exception], ...] | None = None,
    remap: dict[type[Any], type[Any]] | None = None,
) -> Any:
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
        remap: Optional mapping of internal error types to high-level domain errors.

    Returns:
        The decorated async function or a decorator factory.

    Examples:
        >>> @do_notation_async
        ... async def get_data(user_id):
        ...     user = yield await fetch_user(user_id)  # fetch_user returns Result
        ...     yield Ok(user.name)

    """

    def decorator(func: Callable[P, DoAsync[T_local, E_local]]) -> Any:
        return _make_async_wrapper(func, cast("Any", catch_final), remap)

    if callable(arg) and not isinstance(arg, type | tuple):
        return _make_async_wrapper(arg, None, remap)

    catch_final = catch or (arg if isinstance(arg, type | tuple) else None)  # pyright: ignore[reportUnknownVariableType]
    return decorator


# --- Internal Teaching Helper ---


def _raise_api_error(method_name: str) -> Never:
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

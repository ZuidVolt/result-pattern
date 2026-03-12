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
# pyright: reportOverlappingOverload=false
# ruff: noqa: SLF001
# mypy: disable-error-code="no-any-return"

from __future__ import annotations

import asyncio
import inspect
import random
import sys
import time
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Mapping,
)
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Final, Literal, Never, TypeIs, TypeVar, Union, cast, overload

if TYPE_CHECKING:
    from types import TracebackType
    from typing import ParamSpec, Protocol

    from .outcome import Outcome

    P = ParamSpec("P")
    T_ret = TypeVar("T_ret")

    class _CatchDecoratorOrContext[E: Exception](Protocol):
        @overload
        def __call__[T_ret, **P](self, f: Callable[P, T_ret]) -> Callable[P, Result[T_ret, E]]: ...
        @overload
        def __call__[T_ret, **P](
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
    __match_args__ = ("value",)

    def __init__(self, value: T_co) -> None:
        """Initialize an Ok variant.

        Args:
            value: The successful result to wrap.

        """
        object.__setattr__(self, "_value", value)

    @property
    def value(self) -> T_co:
        """Access the success value.

        Note: While accessible on the Ok variant, it is recommended to use
        pattern matching or functional methods for maximum safety.
        """
        return self._value

    def __bool__(self) -> Literal[True]:
        """Truthiness check. Returns True for Ok.

        This enables idiomatic Python conditional narrowing:
        `if res: # type is narrowed to Ok`
        """
        return True

    def __hash__(self) -> int:
        """Explicit hash to prevent collisions with Err variants."""
        return hash((Ok, self._value))

    def __add__(self, other: Any) -> Result[Any, Any]:
        """Support short-circuiting addition (+) operator.

        If both are Ok, returns Ok(a + b). If other is Err, returns other.
        """
        if isinstance(other, Ok):
            return Ok(self._value + other._value)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
        if isinstance(other, Err):
            return other  # pyright: ignore[reportUnknownVariableType]
        return NotImplemented

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

    def map_exc(self, _mapping: Mapping[type[Exception], Any]) -> Ok[T_co]:
        """Ignore the error mapping and return self unchanged.

        Args:
            _mapping: A dictionary mapping exception types to new values.

        Returns:
            The current instance unchanged.

        Examples:
            >>> Ok(10).map_exc({ValueError: ErrorCode.INVALID})
            Ok(10)

        """
        return self

    def cast_types[U, F](self) -> Result[U, F]:
        """Zero-runtime-cost type hint override for strict variance edge cases.

        This allows manually guiding the type checker when it fails to infer
        complex union types correctly.

        Returns:
            The same instance, but with new type parameters for the checker.

        """
        return cast("Result[U, F]", self)

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

    @overload
    def transpose[U](self: Ok[U | None]) -> Ok[U] | None: ...

    @overload
    def transpose(self) -> Ok[T_co] | None: ...

    def transpose(self) -> Ok[Any] | None:
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
        return cast("Ok[Any]", self)

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
            return cast("Any", other)
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
    __match_args__ = ("error",)

    def __init__(self, error: E_co) -> None:
        """Initialize an Err variant.

        Args:
            error: The error state or exception to wrap.

        """
        object.__setattr__(self, "_error", error)

    @property
    def error(self) -> E_co:
        """Access the error state.

        Note: While accessible on the Err variant, it is recommended to use
        pattern matching or functional methods for maximum safety.
        """
        return self._error

    def __bool__(self) -> Literal[False]:
        """Truthiness check. Returns False for Err.

        This enables idiomatic Python conditional narrowing:
        `if res: # type is narrowed to Ok`
        """
        return False

    def __hash__(self) -> int:
        """Explicit hash to prevent collisions with Ok variants."""
        return hash((Err, self._error))

    def __add__(self, other: Any) -> Result[Any, Any]:
        """Support short-circuiting addition (+) operator.

        Always returns self for Err variants, short-circuiting the addition.
        """
        if isinstance(other, (Ok, Err)):
            return self
        return NotImplemented

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

    def map_exc(self, mapping: Mapping[type[Exception], Any]) -> Err[Any]:
        """Replace the error payload if its type exists in the mapping.

        This is ideal for converting raw Python exceptions into domain-specific
        Enums or error codes immediately after they are caught.

        Args:
            mapping: A dictionary mapping exception types to new values.

        Returns:
            A new Err with the mapped value if a match was found, otherwise self.

        Examples:
            >>> # Single mapping to Enum
            >>> Err(ValueError("fail")).map_exc({ValueError: ErrorCode.INVALID})
            Err(<ErrorCode.INVALID: 'invalid'>)

            >>> # Multiple mappings
            >>> mapping = {ValueError: ErrorCode.INVALID, KeyError: ErrorCode.MISSING}
            >>> Err(KeyError("key")).map_exc(mapping)
            Err(<ErrorCode.MISSING: 'missing'>)

        """
        err_type = type(self._error)
        if err_type in mapping:
            return Err(mapping[cast("Any", err_type)])
        return self

    def cast_types[U, F](self) -> Result[U, F]:
        """Zero-runtime-cost type hint override for strict variance edge cases.

        This allows manually guiding the type checker when it fails to infer
        complex union types correctly.

        Returns:
            The same instance, but with new type parameters for the checker.

        """
        return cast("Result[U, F]", self)

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
        return cast("Any", self)


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


def _resolve_mapping(
    exceptions: type[Exception] | tuple[type[Exception], ...] | Mapping[type[Exception], Any],
    map_to: Any = None,
) -> dict[type[Exception], Any]:
    """Normalize exception input into a mapping dictionary."""
    if isinstance(exceptions, Mapping):
        return cast("dict[type[Exception], Any]", dict(exceptions))  # pyright: ignore[reportUnnecessaryCast]

    if isinstance(exceptions, tuple):
        return {exc: map_to if map_to is not None else exc for exc in exceptions}

    return {exceptions: map_to if map_to is not None else exceptions}


@overload
def catch[T, E: Exception, **P](
    exceptions: type[E],
    func: Callable[P, T],
    *,
    map_to: Any = None,
) -> Callable[P, Result[T, Any]]: ...


@overload
def catch[T, E: Exception, **P](
    exceptions: type[E],
    func: Callable[P, Coroutine[Any, Any, T]],
    *,
    map_to: Any = None,
) -> Callable[P, Awaitable[Result[T, Any]]]: ...


@overload
def catch[E: Exception](
    exceptions: type[E],
    func: None = None,
    *,
    map_to: Any = None,
) -> _CatchDecoratorOrContext[E]: ...


@overload
def catch[T, **P](
    exceptions: Mapping[type[Exception], Any],
    func: Callable[P, T],
) -> Callable[P, Result[T, Any]]: ...


@overload
def catch[T, **P](
    exceptions: Mapping[type[Exception], Any],
    func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Awaitable[Result[T, Any]]]: ...


@overload
def catch(
    exceptions: Mapping[type[Exception], Any],
    func: None = None,
) -> _CatchDecoratorOrContext[Exception]: ...


@overload
def catch[T, E1: Exception, E2: Exception, **P](
    exceptions: tuple[type[E1], type[E2]],
    func: Callable[P, T],
    *,
    map_to: Any = None,
) -> Callable[P, Result[T, Any]]: ...


@overload
def catch[T, E1: Exception, E2: Exception, **P](  # type: ignore[overload-overlap] # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]
    exceptions: tuple[type[E1], type[E2]],
    func: Callable[P, Coroutine[Any, Any, T]],
    *,
    map_to: Any = None,
) -> Callable[P, Awaitable[Result[T, Any]]]: ...


@overload
def catch[E1: Exception, E2: Exception](
    exceptions: tuple[type[E1], type[E2]],
    func: None = None,
    *,
    map_to: Any = None,
) -> _CatchDecoratorOrContext[E1 | E2]: ...


@overload
def catch[T, **P](
    exceptions: tuple[type[Exception], ...],
    func: Callable[P, T],
    *,
    map_to: Any = None,
) -> Callable[P, Result[T, Any]]: ...


@overload
def catch[T, **P](
    exceptions: tuple[type[Exception], ...],
    func: Callable[P, Coroutine[Any, Any, T]],
    *,
    map_to: Any = None,
) -> Callable[P, Awaitable[Result[T, Any]]]: ...


@overload
def catch(
    exceptions: tuple[type[Exception], ...],
    func: None = None,
    *,
    map_to: Any = None,
) -> _CatchDecoratorOrContext[Exception]: ...


def catch(  # noqa: C901 # pyright: ignore
    exceptions: Any,
    func: Any = None,
    *,
    map_to: Any = None,
) -> Any:
    """Execute a function and catch specified exceptions into a Result.

    Can be used as a standalone wrapper or as a decorator.

    Args:
        exceptions: An exception type, tuple of types, or mapping of types to values.
        func: Optional function to wrap and execute immediately.
        map_to: Optional value to return in Err if an exception matches.

    Returns:
        A Result if `func` was provided, otherwise a decorator.

    Examples:
        >>> # 1. Simple catch (returns the caught instance)
        >>> @catch(ValueError)
        ... def parse(s: str) -> int:
        ...     return int(s)
        >>> parse("abc")
        Err(ValueError("invalid literal for int()..."))

        >>> # 2. Map single error to Enum using map_to
        >>> @catch(ValueError, map_to=ErrorCode.INVALID)
        ... def parse_safe(s: str) -> int:
        ...     return int(s)
        >>> parse_safe("abc")
        Err(<ErrorCode.INVALID: 'invalid'>)

        >>> # 3. Map multiple errors to Enums using a dictionary
        >>> error_map = {ValueError: ErrorCode.INVALID, KeyError: ErrorCode.MISSING}
        >>> @catch(error_map)
        ... def risky_op(x):
        ...     if x == 0:
        ...         raise ValueError
        ...     if x == 1:
        ...         raise KeyError
        ...     return "ok"
        >>> risky_op(0)
        Err(<ErrorCode.INVALID: 'invalid'>)
        >>> risky_op(1)
        Err(<ErrorCode.MISSING: 'missing'>)

    """
    exc_map = _resolve_mapping(exceptions, map_to)
    catch_tuple = tuple(exc_map.keys())
    # Track if we have an actual mapping value (other than the exception class itself)
    has_mapping = map_to is not None or isinstance(exceptions, Mapping)

    def decorator(f: Callable[P, T]) -> Callable[P, Any]:
        if inspect.iscoroutinefunction(f):

            @wraps(f)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[Any, Any]:
                try:
                    return Ok(await f(*args, **kwargs))
                except catch_tuple as e:
                    mapped = exc_map[type(e)] if has_mapping else e
                    return Err(mapped)

            return async_wrapper

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, Any]:
            __tracebackhide__ = True
            try:
                return Ok(f(*args, **kwargs))
            except catch_tuple as e:
                mapped = exc_map.get(cast("Any", type(e)), e) if has_mapping else e
                return Err(mapped)
            except Exception as e:  # noqa: BLE001
                # Hide the decorator frame from the traceback in modern tools
                # and suppress implementation-detail exception context.
                tb = e.__traceback__
                raise e.with_traceback(tb.tb_next if tb else None) from None

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
            self._ctx = CatchContext(catch_tuple)
            return self._ctx.__enter__()

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> bool:
            if self._ctx is None:
                return False
            exit_res = self._ctx.__exit__(exc_type, exc_val, exc_tb)
            if exit_res and self._ctx.result is not None:
                # Apply mapping if context trapped an error
                err_val = self._ctx.result.err()
                if err_val is not None:
                    mapped = exc_map.get(cast("Any", type(err_val)), err_val)
                    self._ctx.result = Err(mapped)
            return exit_res

    return DecoratorOrContext()


@overload
def catch_call[T, E: Exception](
    exceptions: type[E],
    func: Callable[..., T],
    *args: Any,
    map_to: Any = None,
    **kwargs: Any,
) -> Result[T, Any]: ...


@overload
def catch_call[T, E1: Exception, E2: Exception](
    exceptions: tuple[type[E1], type[E2]],
    func: Callable[..., T],
    *args: Any,
    map_to: Any = None,
    **kwargs: Any,
) -> Result[T, Any]: ...


@overload
def catch_call[T](
    exceptions: Mapping[type[Exception], Any],
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> Result[T, Any]: ...


@overload
def catch_call[T](
    exceptions: tuple[type[Exception], ...],
    func: Callable[..., T],
    *args: Any,
    map_to: Any = None,
    **kwargs: Any,
) -> Result[T, Any]: ...


def catch_call(
    exceptions: Any,
    func: Any,
    *args: Any,
    mapping: Mapping[type[Exception], Any] | None = None,
    map_to: Any = None,
    **kwargs: Any,
) -> Any:
    """Execute a function inline and catch specified exceptions into a Result.

    This allows for clean, single-line expression evaluation without needing
    to define a new function or open a context manager block.

    Args:
        exceptions: An exception type, tuple, or mapping to catch.
        func: The function to execute.
        *args: Positional arguments for the function.
        mapping: Optional explicit mapping dictionary.
        map_to: Optional value to use as the error if an exception matches.
        **kwargs: Keyword arguments for the function.

    Returns:
        Ok(value) if successful, Err(mapped_error) if a specified exception was raised.

    Examples:
        >>> import json
        >>> # 1. Simple catch
        >>> catch_call(json.JSONDecodeError, json.loads, '{"key": "value"}')
        Ok({'key': 'value'})

        >>> # 2. Catch with mapping
        >>> catch_call(json.JSONDecodeError, json.loads, "invalid", map_to="bad_json")
        Err('bad_json')

    """
    exc_map = _resolve_mapping(mapping or exceptions, map_to)
    catch_tuple = tuple(exc_map.keys())
    has_mapping = map_to is not None or mapping is not None or isinstance(exceptions, Mapping)

    try:
        return Ok(func(*args, **kwargs))
    except catch_tuple as e:
        mapped = exc_map.get(type(e), e) if has_mapping else e
        return Err(mapped)


def as_err[E_in: Exception, E_out](
    exception: E_in,
    mapping: type[Exception] | tuple[type[Exception], ...] | Mapping[type[Exception], E_out] | None = None,
    *,
    map_to: E_out | None = None,
) -> Err[E_in | E_out]:
    """Lift a caught exception into an Err variant with optional mapping.

    This pinpoint utility is ideal for manual conversion inside standard
    try/except blocks. It uses the same mapping logic as the @catch decorator.

    Args:
        exception: The exception instance to wrap.
        mapping: Optional exception type, tuple, or dict for transformation.
        map_to: Optional value to use as the error if mapping is a type/tuple.

    Returns:
        An Err variant containing the (potentially mapped) error.

    Examples:
        >>> try:
        ...     raise ValueError("fail")
        ... except Exception as e:
        ...     res = as_err(e, {ValueError: "invalid"})
        >>> res
        Err('invalid')

        >>> # Using map_to
        >>> try:
        ...     raise ValueError("fail")
        ... except Exception as e:
        ...     res = as_err(e, ValueError, map_to="mapped")
        >>> res
        Err('mapped')

    """
    if mapping is None:
        return Err(exception)

    exc_map = _resolve_mapping(mapping, map_to)
    has_mapping = map_to is not None or isinstance(mapping, Mapping)

    if type(exception) in exc_map:
        mapped = exc_map[type(exception)] if has_mapping else exception
        return Err(mapped)

    return Err(exception)


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
        __tracebackhide__ = True
        try:
            gen = func(*args, **kwargs)
            res = next(gen)
            while True:
                if not isinstance(res, Ok | Err):  # pyright: ignore[reportUnnecessaryIsInstance]
                    try:
                        from .outcome import Outcome  # noqa: PLC0415

                        if isinstance(res, Outcome):  # pyright: ignore[reportUnnecessaryIsInstance]
                            fname = getattr(func, "__name__", "<func>")
                            msg = f"Cannot yield Outcome in do_notation. Use 'yield {fname}(...).to_result()' instead."
                            raise TypeError(msg)
                    except ImportError:
                        pass
                    msg = f"do_notation yielded non-Result type: {type(res).__name__}"
                    raise TypeError(msg)  # noqa: TRY301

                if isinstance(res, Err):
                    return _apply_remap(res, remap)
                res = gen.send(res._value)
        except StopIteration as e:
            return Ok(e.value)
        except Exception as e:  # noqa: BLE001
            if catch_types and isinstance(e, catch_types):
                return Err(e)

            # Hide the bind loop frame from the traceback in modern tools
            # and suppress implementation-detail exception context.
            tb = e.__traceback__
            raise e.with_traceback(tb.tb_next if tb else None) from None

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


def _make_async_wrapper[T_local, E_local, **P](  # noqa: C901
    func: Callable[P, DoAsync[T_local, E_local]],
    catch_types: type[Exception] | tuple[type[Exception], ...] | None,
    remap: dict[type[Any], type[Any]] | None = None,
) -> Callable[P, Coroutine[Any, Any, Result[Any, Any]]]:
    """Internal helper to drive the async generator-based bind simulation."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[Any, Any]:
        __tracebackhide__ = True
        try:  # noqa: PLR1702
            gen = func(*args, **kwargs)
            last_val: Result[Any, Any] | None = None
            try:
                res = await anext(gen)
                while True:
                    last_val = res
                    if not isinstance(res, Ok | Err):  # pyright: ignore[reportUnnecessaryIsInstance]
                        try:
                            from .outcome import Outcome  # noqa: PLC0415

                            if isinstance(res, Outcome):  # pyright: ignore[reportUnnecessaryIsInstance]
                                fname = getattr(func, "__name__", "<func>")
                                msg = f"Cannot yield Outcome in do_notation_async. Use 'yield {fname}(...).to_result()' instead."
                                await gen.aclose()
                                raise TypeError(msg)
                        except ImportError:
                            pass
                        msg = f"do_notation_async yielded non-Result type: {type(res).__name__}"
                        await gen.aclose()
                        raise TypeError(msg)

                    if isinstance(res, Err):
                        res_final = _apply_remap(res, remap)
                        await gen.aclose()
                        return res_final
                    res = await gen.asend(res._value)
            except StopAsyncIteration:
                if last_val is None:
                    msg = (
                        "Async do-notation ended without yielding a Result. "
                        "Note: Async generators cannot use 'return value'. "
                        "You must end your function with 'yield Ok(value)'."
                    )
                    raise UnwrapError(Ok(None), msg) from None
                return last_val
        except Exception as e:  # noqa: BLE001
            if catch_types and isinstance(e, catch_types):
                return Err(e)

            # Hide the bind loop frame from the traceback in modern tools
            # and suppress implementation-detail exception context.
            tb = e.__traceback__
            raise e.with_traceback(tb.tb_next if tb else None) from None

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


U_cast = TypeVar("U_cast")
F_cast = TypeVar("F_cast")


class _CastTypesResult[T_inner, E_inner]:
    def __init__(self, owner: Result[T_inner, E_inner]) -> None:
        self._owner = owner

    def __getitem__[U, F](self, _types: Any) -> Callable[[], Result[U, F]]:
        return lambda: cast("Result[U_cast, F_cast]", self._owner)  # type: ignore[valid-type]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]


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

    @property
    def cast_types(self) -> _CastTypesResult[T_co, Any]:
        """Zero-runtime-cost type hint override for strict variance edge cases.

        This allows manually guiding the type checker when it fails to infer
        complex union types correctly.

        Example:
            >>> res = Ok(10).unsafe.cast_types[int, Exception]()

        """
        return _CastTypesResult(self._owner)

    def to_outcome(self) -> Outcome[T_co, None]:
        """Downgrade a strict success into an Outcome.

        Always returns a clean Outcome with no errors.

        Returns:
            A clean Outcome containing the value.

        """
        try:
            from .outcome import Outcome  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "Outcome is not available. use the `result_pattern` pip package and not just the result.py file"
            ) from None

        return Outcome(self._owner._value, None)


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

    @property
    def cast_types(self) -> _CastTypesResult[Any, E_co]:
        """Zero-runtime-cost type hint override for strict variance edge cases.

        This allows manually guiding the type checker when it fails to infer
        complex union types correctly.

        Example:
            >>> res = Err("fail").unsafe.cast_types[int, str]()

        """
        return _CastTypesResult(self._owner)

    def to_outcome[U](self, default: U) -> Outcome[U, E_co]:
        """Downgrade a strict failure into a fault-tolerant Outcome.

        Requires a default value to satisfy the Product Type contract.

        Args:
            default: The fallback value to populate the Outcome.

        Returns:
            A fault-tolerant Outcome containing the default value and the error.

        """
        try:
            from .outcome import Outcome  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "Outcome is not available. use the `result_pattern` pip package and not just the result.py file"
            ) from None

        return Outcome(default, self._owner._error)


def _raise_assertion_error(message: str) -> Any:
    """Internal helper to raise AssertionError and hide this frame from traceback."""
    __tracebackhide__ = True
    try:
        raise AssertionError(message)  # noqa: TRY301
    except AssertionError as e:
        tb = e.__traceback__
        raise e.with_traceback(tb.tb_next if tb else None) from None


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
        frame = sys._getframe(1)
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


_catch_decorator = catch


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

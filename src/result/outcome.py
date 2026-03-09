"""# Outcome: Product Type Error Handling for Modern Python.

This module provides the `Outcome` type, a fault-tolerant product type that
holds both a success value and an optional error state simultaneously.
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportArgumentType=false

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any, NamedTuple, TypeVar, cast, overload

from .result import Err, Ok, Result, _resolve_mapping

T_inner = TypeVar("T_inner")
E_inner = TypeVar("E_inner")


U_cast = TypeVar("U_cast")
F_cast = TypeVar("F_cast")


class Outcome[T, E](NamedTuple):
    """A fault-tolerant Product Type holding a value and an error state.

    'error' can be a single Exception, a Collection of Exceptions, or None.
    Unlike Result, an Outcome always contains a value, even if errors occurred.

    Attributes:
        value: The success or partial success value.
        error: The error state or diagnostic baggage.

    """

    value: T
    error: E | Any | None = None

    @property
    def unsafe(self) -> _OutcomeUnsafe[T, E]:
        """Namespace for operations that bypass standard type safety.

        Examples:
            >>> Outcome(10, None).unsafe.cast_types[int, Exception]()

        """
        return _OutcomeUnsafe(self)

    def has_error(self) -> bool:
        """Check if an error exists, intelligently handling empty collections.

        Returns:
            True if error is not None and (if it's a collection) is not empty.

        Examples:
            >>> Outcome(10, None).has_error()
            False
            >>> Outcome(10, ValueError()).has_error()
            True
            >>> Outcome(10, []).has_error()
            False

        """
        if self.error is None:
            return False

        # Intelligent collection check (excluding str/bytes)
        if isinstance(self.error, Collection) and not isinstance(self.error, str | bytes):
            return len(self.error) > 0

        return True

    def to_result(self) -> Result[T, E]:
        """Convert the Outcome into a strict Sum Type (Result).

        Returns:
            Err(error) if errors exist, otherwise Ok(value).

        Examples:
            >>> Outcome(10, None).to_result()
            Ok(10)
            >>> Outcome(10, ValueError("fail")).to_result()
            Err(ValueError('fail'))

        """
        if self.has_error():
            # We know error is not None here because has_error() checked it
            return Err(self.error)  # type: ignore[arg-type]

        return Ok(self.value)

    def to_ok(self) -> Ok[T]:
        """Discard any error state and return the value wrapped in Ok.

        Returns:
            An Ok variant containing the value.

        """
        return Ok(self.value)

    def map[U_inner](self, func: Callable[[T], U_inner]) -> Outcome[U_inner, E]:
        """Transform the success value while preserving the error state.

        Args:
            func: A function to transform the value.

        Returns:
            A new Outcome with the transformed value and the same error.

        Examples:
            >>> Outcome(10, "warning").map(str)
            Outcome(value='10', error='warning')

        """
        return Outcome(func(self.value), self.error)

    def map_err[F](self, func: Callable[[E], F]) -> Outcome[T, F]:
        """Transform the error payload.

        If the error is a collection, it intelligently maps the function
        over every item in the collection.

        Args:
            func: A function to transform the error(s).

        Returns:
            A new Outcome with the transformed error state.

        Examples:
            >>> Outcome(10, ValueError("fail")).map_err(lambda e: str(e))
            Outcome(value=10, error='fail')

            >>> # Mapping over accumulated errors
            >>> out = Outcome(10, [ValueError("a"), KeyError("b")])
            >>> out.map_err(type)
            Outcome(value=10, error=[<class 'ValueError'>, <class 'KeyError'>])

        """
        if self.error is None:
            return Outcome(self.value, None)

        if isinstance(self.error, list):
            return cast("Any", Outcome(self.value, [func(e) for e in self.error]))  # type: ignore[no-any-return] # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]]

        return Outcome(self.value, func(self.error))  # pyright: ignore[reportArgumentType]

    def map_exc(self, mapping: Mapping[type[Exception], Any]) -> Outcome[T, Any]:
        """Transform specific exception types in the error payload.

        This is ideal for converting raw Python exceptions into domain-specific
        Enums or error codes within a fault-tolerant Outcome.

        Args:
            mapping: A dictionary mapping exception types to new values.

        Returns:
            A new Outcome with the mapped error if a match was found, otherwise self.

        Examples:
            >>> # Single error mapping
            >>> Outcome(10, ValueError("fail")).map_exc({ValueError: ErrorCode.INVALID})
            Outcome(value=10, error=<ErrorCode.INVALID: 'invalid'>)

            >>> # Collection mapping (accumulated diagnostics)
            >>> out = Outcome(10, [ValueError("a"), KeyError("b")])
            >>> out.map_exc({ValueError: "err_val", KeyError: "err_key"})
            Outcome(value=10, error=['err_val', 'err_key'])

        """
        if not self.has_error():
            return self

        # Handle collection of errors
        if isinstance(self.error, Collection) and not isinstance(self.error, str | bytes):  # pyright: ignore[reportUnknownMemberType]
            new_errors = [mapping.get(type(e), e) if isinstance(e, Exception) else e for e in self.error]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            return Outcome(self.value, new_errors)

        if isinstance(self.error, Exception):
            err_type = type(self.error)
            if err_type in mapping:
                return Outcome(self.value, mapping[err_type])

        return self

    def cast_types[U, F](self) -> Outcome[U, F]:
        """Zero-runtime-cost type hint override for strict variance edge cases.

        This allows manually guiding the type checker when it fails to infer
        complex union types correctly.

        Returns:
            The same instance, but with new type parameters for the checker.

        """
        return cast("Outcome[U, F]", self)

    def push_err[E2](self, new_error: E2) -> Outcome[T, E | E2]:
        """Append a new diagnostic error without altering the value.

        If the current error is a list, the new error is appended.
        If it's a single value, both are wrapped in a new list.

        Args:
            new_error: The new error to accumulate.

        Returns:
            A new Outcome with the expanded error state.

        Examples:
            >>> Outcome(10, None).push_err("err1")
            Outcome(value=10, error='err1')
            >>> Outcome(10, "err1").push_err("err2")
            Outcome(value=10, error=['err1', 'err2'])

        """
        if self.error is None:
            return cast("Any", Outcome(self.value, new_error))  # type: ignore[no-any-return]  # ty:ignore[unused-type-ignore-comment, unused-type-ignore-comment, unused-ignore-comment]

        if isinstance(self.error, list):
            return cast("Any", Outcome(self.value, [*self.error, new_error]))  # type: ignore[no-any-return]  # ty:ignore[unused-type-ignore-comment, unused-type-ignore-comment, unused-ignore-comment]

        return cast("Any", Outcome(self.value, [self.error, new_error]))  # type: ignore[no-any-return]  # ty:ignore[unused-type-ignore-comment, unused-type-ignore-comment, unused-ignore-comment]

    def merge[U, E2](self, other: Outcome[U, E2]) -> Outcome[tuple[T, U], E | E2]:
        """Combine two independent outcomes.

        Zips the values into a tuple and concatenates all accumulated errors.

        Args:
            other: Another Outcome to merge with.

        Returns:
            A merged Outcome with tuple value and combined errors.

        Examples:
            >>> out1 = Outcome(1, "e1")
            >>> out2 = Outcome(2, "e2")
            >>> out1.merge(out2)
            Outcome(value=(1, 2), error=['e1', 'e2'])

        """
        errs: list[Any] = []

        # Collect self errors
        if self.error is not None:
            if isinstance(self.error, list):
                errs.extend(self.error)
            else:
                errs.append(self.error)

        # Collect other errors
        if other.error is not None:
            if isinstance(other.error, list):
                errs.extend(other.error)
            else:
                errs.append(other.error)

        final_err = errs or None
        return cast("Any", Outcome((self.value, other.value), final_err))  # type: ignore[no-any-return]  # ty:ignore[unused-type-ignore-comment, unused-type-ignore-comment, unused-ignore-comment]

    def tap_err(self, func: Callable[[E | Any | None], Any]) -> Outcome[T, E]:
        """Execute a side-effect only if an error exists.

        Args:
            func: A function called with the error state if it exists.

        Returns:
            The current Outcome unchanged.

        Examples:
            >>> Outcome(10, "fail").tap_err(print)
            fail
            Outcome(value=10, error='fail')

        """
        if self.has_error():
            func(self.error)
        return self

    def and_then[U, E2](self, func: Callable[[T], Outcome[U, E2]]) -> Outcome[U, E | E2]:
        """The monadic bind for Outcomes.

        Applies the function to the value and concatenates errors from both Outcomes.

        Args:
            func: A function returning a new Outcome.

        Returns:
            A new Outcome with the transformed value and combined errors.

        Examples:
            >>> out = Outcome(10, "e1")
            >>> out.and_then(lambda x: Outcome(x + 1, "e2"))
            Outcome(value=11, error=['e1', 'e2'])

        """
        new_outcome = func(self.value)

        errs: list[Any] = []
        if self.error is not None:
            if isinstance(self.error, list):
                errs.extend(self.error)
            else:
                errs.append(self.error)

        if new_outcome.error is not None:
            if isinstance(new_outcome.error, list):
                errs.extend(new_outcome.error)
            else:
                errs.append(new_outcome.error)

        final_err = errs or None
        return cast("Any", Outcome(new_outcome.value, final_err))  # type: ignore[no-any-return]  # ty:ignore[unused-type-ignore-comment, unused-type-ignore-comment, unused-ignore-comment]

    def or_else[F](self, func: Callable[[E | Any | None], Outcome[T, F]]) -> Outcome[T, F]:
        """Recovery hatch for Outcomes with errors.

        If errors exist, they are passed to the function to attempt recovery.

        Args:
            func: A function taking the current error state and returning a new Outcome.

        Returns:
            The current instance if clean, or the result of func if errors exist.

        Examples:
            >>> out = Outcome(0, "fail")
            >>> out.or_else(lambda _: Outcome(1, None))
            Outcome(value=1, error=None)

        """
        if self.has_error():
            return func(self.error)
        return cast("Outcome[T, F]", self)


@overload
def catch_outcome[T, E: Exception, **P, T_ret](
    exceptions: type[E],
    default: T,
    *,
    map_to: Any = None,
) -> Callable[[Callable[P, T_ret]], Callable[P, Outcome[T | T_ret, Any]]]: ...


@overload
def catch_outcome[T, **P, T_ret](
    exceptions: Mapping[type[Exception], Any],
    default: T,
) -> Callable[[Callable[P, T_ret]], Callable[P, Outcome[T | T_ret, Any]]]: ...


@overload
def catch_outcome[T, **P, T_ret](
    exceptions: tuple[type[Exception], ...],
    default: T,
    *,
    map_to: Any = None,
) -> Callable[[Callable[P, T_ret]], Callable[P, Outcome[T | T_ret, Any]]]: ...


def catch_outcome(
    exceptions: Any,
    default: Any,
    *,
    map_to: Any = None,
) -> Any:
    """Catch exceptions and return an Outcome with a fallback value.

    Requires a 'default' value to populate the Outcome payload on crash.

    Args:
        exceptions: An exception type, tuple of types, or mapping of types to values.
        default: The fallback value to use if an exception is caught.
        map_to: Optional value to use as the error if an exception matches.

    Returns:
        A decorator that wraps a function to return an Outcome.

    Examples:
        >>> # 1. Simple catch with default
        >>> @catch_outcome(ValueError, default=0)
        ... def parse(s: str) -> int:
        ...     return int(s)
        >>> parse("abc")
        Outcome(value=0, error=ValueError(...))

        >>> # 2. Map single error using map_to
        >>> @catch_outcome(ValueError, default=-1, map_to=ErrorCode.INVALID)
        ... def risky_parse(s: str) -> int:
        ...     return int(s)
        >>> risky_parse("abc")
        Outcome(value=-1, error=<ErrorCode.INVALID: 'invalid'>)

        >>> # 3. Map multiple errors using a dictionary
        >>> error_map = {ValueError: ErrorCode.INVALID, KeyError: ErrorCode.MISSING}
        >>> @catch_outcome(error_map, default=None)
        ... def complex_op(x):
        ...     if x == 0:
        ...         raise ValueError
        ...     if x == 1:
        ...         raise KeyError
        ...     return "ok"
        >>> complex_op(0)
        Outcome(value=None, error=<ErrorCode.INVALID: 'invalid'>)

    """
    exc_map = _resolve_mapping(exceptions, map_to)
    catch_tuple = tuple(exc_map.keys())
    # Track if we have an actual mapping value (other than the exception class itself)
    has_mapping = map_to is not None or isinstance(exceptions, Mapping)

    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Outcome[Any, Any]:
            __tracebackhide__ = True
            try:
                return Outcome(func(*args, **kwargs), None)
            except catch_tuple as e:
                # If we have an explicit mapping, use it. Otherwise, use the instance 'e'.
                mapped = exc_map[type(e)] if has_mapping else e
                return Outcome(default, mapped)
            except Exception as e:  # noqa: BLE001
                # Hide the decorator frame from the traceback in modern tools
                # and suppress implementation-detail exception context.
                tb = e.__traceback__
                raise e.with_traceback(tb.tb_next if tb else None) from None

        return wrapper

    return decorator


class _CastTypesOutcome[T_inner, E_inner]:
    def __init__(self, owner: Outcome[T_inner, E_inner]) -> None:
        self._owner = owner

    def __getitem__[U, F](self, _types: Any) -> Callable[[], Outcome[U, F]]:
        return lambda: cast("Outcome[U_cast, F_cast]", self._owner)  # type: ignore[valid-type]  # ty:ignore[unused-type-ignore-comment, unused-ignore-comment]


@dataclass(frozen=True, slots=True)
class _OutcomeUnsafe[T_inner, E_inner]:
    """Namespace for potentially unsafe operations on Outcome."""

    _owner: Outcome[T_inner, E_inner]

    @property
    def cast_types(self) -> _CastTypesOutcome[T_inner, E_inner]:
        """Zero-runtime-cost type hint override for strict variance edge cases.

        This allows manually guiding the type checker when it fails to infer
        complex union types correctly.

        Example:
            >>> res = Outcome(10, None).unsafe.cast_types[int, str]()

        """
        return _CastTypesOutcome(self._owner)

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from result import (
    Err,
    Ok,
    OkErr,
    Result,
    collecting,
    safe_resource,
    scoped_catch,
)

if TYPE_CHECKING:
    from result.adapters import Collector, ScopedCatch

# --- Part 1: Basic Syntax and Usability ---


def test_safe_resource_success() -> None:
    """Verify safe_resource with a successful context manager."""
    expected_val = 42

    class MockCM:
        def __enter__(self) -> int:
            return expected_val

        def __exit__(self, *args: Any) -> None:
            pass

    res: Result[int, Exception]
    with safe_resource(MockCM()) as res:  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(res, OkErr)
        assert res == Ok(expected_val)


def test_safe_resource_fail_enter() -> None:
    """Verify safe_resource captures exceptions in __enter__."""

    class FailCM:
        def __enter__(self) -> Any:
            msg = "enter fail"
            raise ValueError(msg)

        def __exit__(self, *args: Any) -> None:
            pass

    res: Result[Any, Exception]
    with safe_resource(FailCM()) as res:  # pyright: ignore[reportUnknownVariableType]
        assert isinstance(res, OkErr)
        assert res.is_err()
        assert str(res.err()) == "enter fail"


def test_collecting_usability() -> None:
    """Verify basic usage of collecting()."""
    val1 = 10
    val2 = 20

    def parse_int(s: str) -> Result[int, str]:
        try:
            return Ok(int(s))
        except ValueError:
            return Err(f"invalid: {s}")

    col: Collector[str]
    with collecting[str]() as col:
        v1 = col.add(parse_int("10"))
        v2 = col.add(parse_int("abc"))
        v3 = col.add(parse_int("20"))
        v4 = col.add(parse_int("xyz"))

    assert v1 == val1
    assert v2 is None
    assert v3 == val2
    assert v4 is None
    assert col.ok is False
    assert col.errors == ["invalid: abc", "invalid: xyz"]


def test_scoped_catch_success() -> None:
    """Verify scoped_catch with successful execution."""
    val = 100
    scope: ScopedCatch[int, str]
    with scoped_catch[int, str]() as scope:
        scope.on(ValueError, "bad value")
        scope.set(val)

    assert scope.result == Ok(val)


def test_scoped_catch_routing() -> None:
    """Verify scoped_catch routes different exceptions correctly."""
    scope: ScopedCatch[int, str]
    with scoped_catch[int, str]() as scope:
        scope.on(ValueError, "VALUE_ERROR")
        scope.on(KeyError, "KEY_ERROR")
        msg = "missing"
        raise KeyError(msg)

    assert scope.result == Err("KEY_ERROR")


# --- Part 2: Internal Property and Integration Testing ---


def test_safe_resource_cleanup_logic() -> None:
    """Ensure cleanup (__exit__) is only called if __enter__ succeeded."""
    cleanup_called = False

    class TrackingCM:
        def __enter__(self) -> TrackingCM:
            return self

        def __exit__(self, *args: Any) -> None:
            nonlocal cleanup_called
            cleanup_called = True

    res: Result[TrackingCM, Exception]
    with safe_resource(TrackingCM()) as res:  # pyright: ignore[reportUnknownVariableType]
        assert res.is_ok()
    assert cleanup_called is True

    cleanup_called = False

    class FailEnterCM:
        def __enter__(self) -> Any:
            raise RuntimeError

        def __exit__(self, *args: Any) -> None:
            nonlocal cleanup_called
            cleanup_called = True

    res_fail: Result[Any, Exception]
    with safe_resource(FailEnterCM()) as res_fail:  # pyright: ignore[reportUnknownVariableType]
        assert res_fail.is_err()
    assert cleanup_called is False


def test_collecting_integration_validation() -> None:
    """Test collecting() in a realistic validation scenario."""
    expected_age = 30

    def validate_user(data: dict[str, Any]) -> Result[dict[str, Any], list[str]]:
        col: Collector[str]
        with collecting[str]() as col:
            name = col.add(Ok(data["name"]) if "name" in data else Err("missing name"))
            age = col.add(Ok(data["age"]) if "age" in data and data["age"] > 0 else Err("invalid age"))

        if col.ok:
            return Ok({"name": name, "age": age})
        return Err(col.errors)

    res = validate_user({"name": "Alice", "age": expected_age})
    assert res == Ok({"name": "Alice", "age": expected_age})

    res = validate_user({})
    assert res == Err(["missing name", "invalid age"])


def test_scoped_catch_unhandled_exception() -> None:
    """Ensure unhandled exceptions bubble up from scoped_catch."""
    scope: ScopedCatch[int, str]
    with pytest.raises(RuntimeError, match="bubble"), scoped_catch[int, str]() as scope:
        scope.on(ValueError, "mapped")
        msg = "bubble"
        raise RuntimeError(msg)


def test_scoped_catch_inheritance() -> None:
    """Verify that scoped_catch respects exception inheritance."""

    class MyError(ValueError):
        pass

    scope: ScopedCatch[int, str]
    with scoped_catch[int, str]() as scope:
        scope.on(ValueError, "BASE_MAPPED")
        msg = "child"
        raise MyError(msg)

    assert scope.result == Err("BASE_MAPPED")

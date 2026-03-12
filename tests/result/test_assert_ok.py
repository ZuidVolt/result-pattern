import pytest

from result import Err, Ok, assert_ok


def test_assert_ok_function_success() -> None:
    """Verify assert_ok as a function returns value on Ok."""
    expected = 42
    res = Ok(expected)
    val = assert_ok(res)
    assert val == expected


def test_assert_ok_function_failure() -> None:
    """Verify assert_ok as a function raises AssertionError on Err."""
    res = Err("something went wrong")
    with pytest.raises(AssertionError, match="assert_ok failed: something went wrong"):
        assert_ok(res)


def test_assert_ok_context_manager_success() -> None:
    """Verify assert_ok as a context manager allows multiple checks."""
    v1, v2 = 10, 20
    with assert_ok("Critical operations") as ctx:
        val1 = ctx.check(Ok(v1))
        val2 = ctx.check(Ok(v2))

    assert val1 == v1
    assert val2 == v2


def test_assert_ok_context_manager_failure() -> None:
    """Verify assert_ok as a context manager raises AssertionError with custom message."""
    with (
        pytest.raises(AssertionError, match="Database operations: connection refused"),
        assert_ok("Database operations") as ctx,
    ):
        ctx.check(Ok("connected"))
        ctx.check(Err("connection refused"))
        # This line should never be reached
        ctx.check(Ok("done"))


def test_assert_ok_nested() -> None:
    """Verify assert_ok can be nested."""
    with assert_ok("Outer") as outer:
        val_outer = outer.check(Ok("outer_ok"))
        with assert_ok("Inner") as inner:
            val_inner = inner.check(Ok("inner_ok"))

    assert val_outer == "outer_ok"
    assert val_inner == "inner_ok"


def test_assert_ok_automatic_scanning() -> None:
    """Verify that assert_ok automatically detects Err assignments."""
    with (
        pytest.raises(AssertionError, match="Auto-check: boom"),
        assert_ok("Auto-check"),
    ):
        # This assignment should be caught by the tracer
        res = Err("boom")
        # This line should never be reached
        print(res)


def test_assert_ok_automatic_scanning_ignore_existing() -> None:
    """Verify that assert_ok ignores Err variables that existed before the block."""
    existing_err = Err("already here")

    with assert_ok("Should pass"):
        # Accessing an existing Err shouldn't trigger if it's not reassigned
        _ = existing_err
        x = Ok(1)
        assert x.is_ok()

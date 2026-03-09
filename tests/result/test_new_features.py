import pytest

from result import Err, Ok, Result, partition_map


def test_partition_map_basic() -> None:
    """Verify partition_map separates items while preserving context."""

    def parse(s: str) -> Result[int, str]:
        return Ok(int(s)) if s.isdigit() else Err(f"not a digit: {s}")

    items = ["1", "a", "2", "b"]
    oks, errs = partition_map(items, parse)

    assert oks == [("1", 1), ("2", 2)]
    assert errs == [("a", "not a digit: a"), ("b", "not a digit: b")]


def test_named_pattern_matching() -> None:
    """Verify keyword pattern matching works for Ok and Err."""
    expected_val = 10
    expected_err = "fail"
    res_ok: Result[int, str] = Ok(expected_val)
    res_err: Result[int, str] = Err(expected_err)

    # We match against the union type to test the pattern matching logic
    # but we don't include unreachable cases to please basedpyright.
    match res_ok:
        case Ok(value=val):
            assert val == expected_val
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            pass

    match res_err:
        case Err(error=err):
            assert err == expected_err
        case _:  # pyright: ignore[reportUnnecessaryComparison]
            pass


def test_direct_access_educational_safeguard() -> None:
    """Verify that direct access to .value/error on Result union still warns/fails."""
    # Actually, because we added .value to Ok and .error to Err,
    # if you have a Result and it's an Err, calling .value will go to __getattr__.

    res_err: Result[int, str] = Err("fail")
    with pytest.raises(AttributeError, match=r"Result API Warning: Direct access to '.value'"):
        _ = res_err.value

    res_ok: Result[int, str] = Ok(10)
    with pytest.raises(AttributeError, match=r"Result API Warning: Direct access to '.error'"):
        _ = res_ok.error


def test_nested_result_matching() -> None:
    """Verify matching nested results with keywords."""
    magic_val = 42
    nested: Result[Result[int, str], str] = Ok(Ok(magic_val))

    match nested:
        case Ok(value=Ok(value=val)):
            assert val == magic_val
        case _:
            pass

    nested_err: Result[Result[int, str], str] = Ok(Err("inner"))
    match nested_err:
        case Ok(value=Err(error=err)):
            assert err == "inner"
        case _:
            pass

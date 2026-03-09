# ruff: noqa: PLR2004


from typing import Any

from result import Err, Ok, is_err
from result.outcome import Outcome, as_outcome, catch_outcome


def test_outcome_native_unpacking() -> None:
    """Verify Outcome unpacks natively like a Go/Odin return value."""

    def parse_stmt(s: str) -> Outcome[str, str]:
        if not s:
            return Outcome("", "Empty statement")
        return Outcome(f"AST({s})")

    # Unpack like a tuple
    node, err = parse_stmt("x = 10")
    assert node == "AST(x = 10)"
    assert err is None

    bad_node, bad_err = parse_stmt("")
    assert not bad_node
    assert bad_err == "Empty statement"


def test_outcome_to_result_conversion() -> None:
    """Verify Outcome -> Result conversion (Sum Type Bridge)."""
    # 1. Clean success
    out_ok: Outcome[int, Any] = Outcome(100)
    assert out_ok.to_result() == Ok(100)

    # 2. Singular error
    out_err = Outcome(0, ValueError("fatal"))
    assert out_err.to_result() == Err(out_err.error)

    # 3. Accumulated diagnostics (Collection)
    # has_error() handles empty vs non-empty lists
    out_warns = Outcome({"ast": "node"}, ["Missing semicolon", "Unused var"])
    assert out_warns.has_error() is True
    assert is_err(out_warns.to_result())

    out_empty_warns: Outcome[dict[str, str], list[str]] = Outcome({"ast": "node"}, [])
    assert out_empty_warns.has_error() is False
    assert out_empty_warns.to_result() == Ok({"ast": "node"})


def test_outcome_functional_orchestration() -> None:
    """Verify Outcome functional methods."""
    out = Outcome(10, "warn")
    # Mapping transforms value but preserves error
    mapped: Outcome[str, str] = out.map(str)
    assert mapped.value == "10"
    assert mapped.error == "warn"

    # to_ok() discards error
    assert out.to_ok() == Ok(10)


def test_outcome_accumulators() -> None:
    """Verify Outcome error accumulation logic."""
    # 1. push_err
    out: Outcome[int, Any] = Outcome(10, None)
    out2: Outcome[int, Any] = out.push_err("e1")
    assert out2.error == "e1"
    out3: Outcome[int, Any] = out2.push_err("e2")
    assert out3.error == ["e1", "e2"]

    # 2. merge
    res: Outcome[tuple[int, int], Any] = Outcome(1, "err1").merge(Outcome(2, "err2"))
    assert res == Outcome((1, 2), ["err1", "err2"])


def test_outcome_functional_chaining() -> None:
    """Verify Outcome monadic bind and recovery."""
    # 1. and_then
    out: Outcome[int, str] = Outcome(10, "e1")
    res: Outcome[int, Any] = out.and_then(lambda x: Outcome(x + 1, "e2"))
    assert res == Outcome(11, ["e1", "e2"])

    # 2. or_else
    clean: Outcome[int, Any] = Outcome(10, None)
    assert clean.or_else(lambda _: Outcome(0, None)) == clean  # pyright: ignore[reportUnknownLambdaType]

    failed: Outcome[int, str] = Outcome(0, "fail")
    assert failed.or_else(lambda _: Outcome(1, None)) == Outcome(1, None)  # pyright: ignore[reportUnknownLambdaType]


def test_outcome_side_effects() -> None:
    """Verify tap_err for Outcome."""
    errors: list[Any] = []
    out: Outcome[int, str] = Outcome(10, "err")
    out.tap_err(errors.append)
    assert errors == ["err"]

    clean: Outcome[int, Any] = Outcome(10, None)
    clean.tap_err(errors.append)
    assert len(errors) == 1


def test_result_to_outcome_bridge() -> None:
    """Verify unsafe bridge from Result to Outcome."""
    # Success path
    assert Ok(10).unsafe.to_outcome() == Outcome(10, None)

    # Failure path with default
    err_inst = ValueError("fail")
    err_res = Err(err_inst)
    out = err_res.unsafe.to_outcome(default=0)
    assert out.value == 0
    assert out.error is err_inst


def test_outcome_map_exc() -> None:
    """Verify Outcome.map_exc transforming error payloads."""
    # 1. Single error mapping
    out = Outcome(10, ValueError("fail"))
    assert out.map_exc({ValueError: "invalid"}) == Outcome(10, "invalid")

    # 2. Collection mapping
    err_c = RuntimeError("c")
    out_coll = Outcome(10, [ValueError("a"), KeyError("b"), err_c])
    mapped = out_coll.map_exc({ValueError: "err_val", KeyError: "err_key"})
    assert mapped.value == 10
    # Use direct reference for unmapped error to avoid instance equality issues
    assert mapped.error == ["err_val", "err_key", err_c]


def test_catch_outcome_api() -> None:
    """Verify catch_outcome decorator behavior."""

    # 1. Basic success/fail with default
    @catch_outcome(ValueError, default=0)
    def parse(s: str) -> int:
        return int(s)

    assert parse("10") == Outcome(10, None)
    res_err = parse("abc")
    assert res_err.value == 0
    assert isinstance(res_err.error, Exception)

    # 2. With mapping
    @catch_outcome({ValueError: "invalid"}, default=-1)
    def risky(s: str) -> int:
        return int(s)

    assert risky("abc") == Outcome(-1, "invalid")


def test_as_outcome_pinpoint() -> None:
    """Verify as_outcome for manual exception lifting."""
    e = ValueError("fail")
    # 1. Simple lift
    assert as_outcome(e, default=0) == Outcome(0, e)

    # 2. Mapping
    assert as_outcome(e, default=10, mapping={ValueError: "mapped"}) == Outcome(10, "mapped")

    # 3. Mismatch
    assert as_outcome(e, default=1, mapping={KeyError: "err"}) == Outcome(1, e)

    # 4. map_to
    assert as_outcome(e, default=-1, mapping=ValueError, map_to="fail") == Outcome(-1, "fail")

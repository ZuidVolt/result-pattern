import asyncio
from typing import Any

import pytest

from result import Err, Ok, catch_boundary, catch_instance, is_err

# --- Classes for testing ---


class RawClient:
    def sync_ok(self, x: int) -> int:
        return x

    def sync_fail(self) -> None:
        raise ValueError("sync fail")

    async def async_ok(self, x: int) -> int:
        await asyncio.sleep(0)
        return x

    async def async_fail(self) -> None:
        await asyncio.sleep(0)
        raise ValueError("async fail")

    def _private(self) -> str:
        return "private"


@catch_boundary(ValueError)
class SafeClient:
    def ok(self, x: int) -> Any:
        return x

    def fail(self) -> Any:
        raise ValueError("boundary fail")

    async def aok(self, x: int) -> Any:
        return x

    async def afail(self) -> Any:
        raise ValueError("async boundary fail")


# --- Tests ---


def test_catch_boundary_sync() -> None:
    client: Any = SafeClient()
    assert client.ok(10) == Ok(10)
    res: Any = client.fail()
    assert is_err(res)
    # Check that it's a ValueError with the right message
    err: Any = res.err()
    assert isinstance(err, ValueError)
    assert str(err) == "boundary fail"


@pytest.mark.asyncio
async def test_catch_boundary_async() -> None:
    client: Any = SafeClient()
    assert await client.aok(20) == Ok(20)
    res: Any = await client.afail()
    assert is_err(res)
    err: Any = res.err()
    assert isinstance(err, ValueError)
    assert str(err) == "async boundary fail"


def test_catch_boundary_mapping() -> None:
    @catch_boundary(ValueError, map_to="mapped")
    class MappedClient:
        def fail(self) -> Any:
            raise ValueError

    assert MappedClient().fail() == Err("mapped")


def test_catch_instance_sync() -> None:
    raw = RawClient()
    safe: Any = catch_instance(raw, ValueError)

    assert safe.sync_ok(42) == Ok(42)
    res: Any = safe.sync_fail()
    assert is_err(res)  # ty:ignore[invalid-argument-type]
    err: Any = res.err()  # ty:ignore[unresolved-attribute]
    assert isinstance(err, ValueError)
    assert str(err) == "sync fail"

    # Verify private methods are not wrapped
    assert safe._private() == "private"  # noqa: SLF001


@pytest.mark.asyncio
async def test_catch_instance_async() -> None:
    raw = RawClient()
    safe: Any = catch_instance(raw, ValueError)

    assert await safe.async_ok(100) == Ok(100)
    res: Any = await safe.async_fail()
    assert is_err(res)  # ty:ignore[invalid-argument-type]
    err: Any = res.err()  # ty:ignore[unresolved-attribute]
    assert isinstance(err, ValueError)
    assert str(err) == "async fail"


def test_catch_instance_repr() -> None:
    raw = RawClient()
    safe = catch_instance(raw, ValueError)
    assert "catch_instance" in repr(safe)

from typing import Any

import pytest
from cachetools import LRUCache, TTLCache

from .cache import cache_set, lfx_cache


@pytest.fixture(autouse=True)
def clear_cache_set():
    before = set(cache_set.keys())
    yield
    for key in set(cache_set.keys()) - before:
        del cache_set[key]


def _make_cache_key(host: str, port: int):
    return hash(f"connection_pool:{host}:{port}")


@lfx_cache(cache_factory=lambda: LRUCache(maxsize=1024), key=_make_cache_key)
def get_pool(host: str, port: int) -> dict[str, Any]:
    return {"host": host, "port": port}


def test_cache():
    a = get_pool("localhost", 9669)
    b = get_pool("localhost", 9669)
    assert a == b
    assert a is b
    assert a["host"] == "localhost"
    assert a["port"] == 9669
    assert b["host"] == "localhost"
    assert b["port"] == 9669


def test_cache_different():
    a = get_pool("host_a", 1)
    b = get_pool("host_b", 2)
    assert a != b
    assert a is not b
    assert a["host"] == "host_a"
    assert a["port"] == 1
    assert b["host"] == "host_b"
    assert b["port"] == 2


def test_cache_key():
    call_count = 0

    def make_key(x: int, y: int):
        return hash(f"{x}:{y}")

    @lfx_cache(cache_factory=lambda: LRUCache(maxsize=64), key=make_key)
    def computed(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        return x + y

    assert computed(1, 2) == 3
    assert computed(1, 2) == 3
    assert call_count == 1
    assert computed(2, 3) == 5
    assert call_count == 2


def test_cache_key_static_cahce_key():
    call_count = 0

    def make_key(_x: int, _y: int):
        return hash("static")

    @lfx_cache(cache_factory=lambda: LRUCache(maxsize=64), key=make_key)
    def computed(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        return x + y

    assert computed(1, 2) == 3
    assert call_count == 1
    assert computed(99, 99) == 3
    assert call_count == 1


def test_cache_different_cache_type():
    def make_key(_: int):
        return hash("abc")

    @lfx_cache(cache_factory=lambda: LRUCache(maxsize=8), key=make_key)
    def f_a(x: int) -> str:
        return f"a_{x}"

    @lfx_cache(cache_factory=lambda: LRUCache(maxsize=8), key=make_key)
    def f_b(x: int) -> str:
        return f"b_{x}"

    assert f_a(1) == "a_1"
    assert f_b(1) == "b_1"


def test_ttl():
    @lfx_cache(cache_factory=lambda: TTLCache(maxsize=64, ttl=300), key=lambda x: x)
    def f(x: int) -> int:
        return x * 2

    assert f(3) == 6
    assert f(3) == 6

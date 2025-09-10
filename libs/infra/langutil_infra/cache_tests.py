import time
from typing import Any

from cachetools import cached

from .cache import TTICache


def test_tti():
    def ttu(_key: Any, _value: Any, now: float) -> float:
        return now + 2  # 2 seconds from now

    cache = TTICache(maxsize=5, ttl=2, ttu=ttu)

    cache["a"] = 1
    cache["b"] = 2

    assert list(cache.keys()) == ["a", "b"]

    time.sleep(1)
    cache["a"]
    assert list(cache.keys()) == ["b", "a"]

    time.sleep(1)
    assert list(cache.keys()) == ["a"]


def test_tti2():
    def ttu(_key: Any, _value: Any, now: float) -> float:
        return now + 1  # 1 seconds from now

    cache = TTICache(maxsize=5, ttl=1, ttu=ttu)

    cache["a"] = 1
    cache["b"] = 2

    time.sleep(1)
    assert list(cache.keys()) == []


def test_tti3():
    call_count = 0

    def ttu(_key: Any, _value: Any, now: float) -> float:
        return now + 1  # 1 seconds from now

    cache = TTICache(maxsize=5, ttl=1, ttu=ttu)

    @cached(cache=cache)
    def f():
        nonlocal call_count
        call_count = call_count + 1
        return call_count

    f()
    assert call_count == 1
    f()
    assert call_count == 1

    time.sleep(1)
    f()
    assert call_count == 2

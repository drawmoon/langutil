import hashlib
import inspect
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from cachetools import LRUCache, TLRUCache, TTLCache, cached
from langutil_infra.cache import TTICache

CacheLike = (
    LRUCache[Any, Any] | TLRUCache[Any, Any] | TTLCache[Any, Any] | TTICache[Any, Any]
)


def _ttu(_key: Any, _value: Any, now: float) -> float:
    return now + timedelta(hours=6).total_seconds()


cache_set = TLRUCache(maxsize=1024, ttu=_ttu)

R = TypeVar("R")
P = ParamSpec("P")


def lfx_cache(cache_factory: Callable[[], CacheLike], key: Callable[..., int]):
    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        module = _make_module_fingerprint(function)
        if module not in cache_set:
            cache_set[module] = cache_factory()
        cache = cache_set[module]
        return cached(cache=cache, key=key)(function)

    return decorator


def _make_module_fingerprint(fn: Callable[..., Any]):
    module = inspect.getmodule(fn)
    file_path = getattr(module, "__file__", None)

    if file_path and Path(file_path).exists():
        try:
            source_content = inspect.getsource(fn).encode("utf-8")
            return hashlib.sha256(source_content).hexdigest()
        except (OSError, TypeError):
            pass

    # NOTE: Code dynamically created via exec/eval does not have accessible source code text.
    # Therefore, we use a hash based on the bytecode and other attributes to robustly identify functions.
    try:
        code = fn.__code__
        features = [
            code.co_code,
            str(code.co_consts),
            str(code.co_names),
            fn.__qualname__,
        ]
        fingerprint = "|".join(map(str, features)).encode("utf-8")
        return hashlib.sha256(fingerprint).hexdigest()
    except Exception:
        return hashlib.sha256(repr(fn).encode()).hexdigest()

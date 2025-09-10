import time
from collections.abc import Callable
from typing import Any, TypeVar

from cachetools import TTLCache

# TTU: (key, value, now) -> expiration_time
TTUCallable = Callable[[Any, Any, float], float]

K = TypeVar("K")
V = TypeVar("V")


class TTICache(TTLCache[K, V]):
    def __init__(
        self,
        maxsize: int,
        ttl: float,
        ttu: TTUCallable,
        timer: Callable[[], float] = time.monotonic,
        getsizeof: Callable[[Any], int] | None = None,
    ) -> None:
        super().__init__(maxsize, ttl, timer, getsizeof)
        self.__ttu = ttu

    def __getitem__(
        self,
        key: K,
        cache_getitem: Callable[..., V] = TTLCache.__getitem__,  # type: ignore[assignment]
    ) -> V:
        # 1. Call the superclass to get the value (if expired, superclass will raise KeyError or handle deletion)
        value = cache_getitem(self, key)

        # 2. If the value is retrieved, it has not absolutely expired; now execute TTU (Time To Use) logic
        now = self.timer()
        new_expiration = self.__ttu(key, value, now)

        # 3. Update the expiration time for this key
        # cachetools internally uses self._TTLCache__links to store expiration info
        # We need to update the link object's expires property and move it to the end of the queue (expire last)
        link = self._TTLCache__links[key]  # pyright: ignore[reportAttributeAccessIssue]
        link.expires = new_expiration

        # Move the link from its current position to the tail of the doubly-linked list
        # This ensures that the expiration order matches the linked list order
        self._update_link(link)

        return value

    def _update_link(self, link: TTLCache._Link) -> None:  # pyright: ignore[reportAttributeAccessIssue]
        """Move the link to the end of the doubly linked list, maintaining expiration order"""
        root = self._TTLCache__root  # pyright: ignore[reportAttributeAccessIssue]
        links = self._TTLCache__links  # pyright: ignore[reportAttributeAccessIssue]
        # 1. Unlink from current position
        link.prev.next = link.next
        link.next.prev = link.prev
        # 2. Insert at tail (before root)
        link.next = root
        link.prev = root.prev
        link.prev.next = link
        root.prev = link
        # 3. Keep OrderedDict in sync with linked list order
        links.move_to_end(link.key)

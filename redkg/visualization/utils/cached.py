"""Cached decorator function."""

from functools import lru_cache
from typing import Callable, Any


def cached(maxsize: int = 32, typed: bool = True) -> Callable[[Callable], Callable]:
    """Make function cached with LRU Cache."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Cache with LRU cache
        cache = lru_cache(maxsize=maxsize, typed=typed)

        @cache
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    return decorator

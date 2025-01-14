"""Frozen dataset for immutable objects."""

from dataclasses import dataclass
from typing import Type, TypeVar, Callable

T = TypeVar('T')  # Generic type for class


def frozen_dataclass(cls: Type[T]) -> Type[T]:
    """Frozen dataclass decorator for dataclass-based immutable enums."""
    return dataclass(frozen=True)(cls)


def reference(cls: Type[T]) -> Type[T]:
    """Frozen dataclass decorator for dataclass-based immutable enums."""
    return frozen_dataclass(cls)

"""Frozen dataset for immutable objects."""

from dataclasses import dataclass


def frozen_dataclass(cls):
    """Frozen dataclass decorator for dataclass based immutable enums."""
    return dataclass(frozen=True)(cls)


def reference(cls):
    """Frozen dataclass decorator for dataclass based immutable enums."""
    return frozen_dataclass(cls)

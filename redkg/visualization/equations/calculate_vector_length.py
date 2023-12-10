"""Calculate vertex length module."""

import math


def calculate_vector_length(vector: list) -> float:
    """Calculate vertex length function."""
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

"""Calculate tangent radian module."""

import math


def calculate_common_tangent_radian(r1: int, r2: int, d: float) -> float:
    """Calculate tangent radian function."""
    alpha = math.acos(abs(r2 - r1) / d)
    alpha = alpha if r1 > r2 else math.pi - alpha
    return alpha

"""Convert radians from atan."""

import math


def calculate_radian_from_atan(x: int, y: int) -> float:
    """Convert radians from atan function."""
    if x == 0:
        return math.pi / 2 if y > 0 else 3 * math.pi / 2
    if y == 0:
        return 0 if x > 0 else math.pi
    r = math.atan(y / x)
    if x > 0 and y > 0:
        return r
    elif x > 0 > y:
        return r + 2 * math.pi
    elif x < 0 < y:
        return r + math.pi
    else:
        return r + math.pi

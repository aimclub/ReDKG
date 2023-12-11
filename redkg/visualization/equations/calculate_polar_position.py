"""Calculate polar position module."""

import math
from typing import Any

import numpy as np


def calculate_polar_position(r: int, theta: float, start_point: Any) -> Any:
    """Calculate polar position function."""
    return np.array([r * math.cos(theta), r * math.sin(theta)]) + start_point

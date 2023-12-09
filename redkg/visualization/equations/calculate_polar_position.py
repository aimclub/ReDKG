"""Calculate polar position module."""

import math

import numpy as np


def calculate_polar_position(r, theta, start_point):
    """Calculate polar position function."""
    return np.array([r * math.cos(theta), r * math.sin(theta)]) + start_point

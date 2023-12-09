"""Calculate direction module."""

import numpy as np


def calculate_direction(direction):
    """Calculate direction function."""
    return direction / np.linalg.norm(direction)

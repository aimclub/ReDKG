"""Calculate direction module."""

import numpy as np


def calc_direction(direction):
    """Calculate direction function."""
    return direction / np.linalg.norm(direction)

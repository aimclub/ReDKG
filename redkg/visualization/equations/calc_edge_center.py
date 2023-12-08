"""Calculate edge center module."""

import numpy as np


def calc_edge_center(H, position):
    """Calculate edge center function."""
    return np.matmul(H.T, position) / H.sum(axis=0).reshape(-1, 1)

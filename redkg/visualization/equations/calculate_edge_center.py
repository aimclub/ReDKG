"""Calculate edge center module."""

from typing import Any

import numpy as np


def calculate_edge_center(H: Any, position: Any) -> Any:
    """Calculate edge center function."""
    return np.matmul(H.T, position) / H.sum(axis=0).reshape(-1, 1)

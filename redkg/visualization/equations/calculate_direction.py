"""Calculate direction module."""

from typing import Any, Union

import numpy as np


def calculate_direction(direction: Any) -> Union[float, np.ndarray]:
    """Calculate direction function."""
    return direction / np.linalg.norm(direction)

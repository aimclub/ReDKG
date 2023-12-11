"""Calculate safe div module."""
from typing import Any

import numpy as np

from redkg.visualization.config.parameters.defaults import Defaults


def calculate_safe_div(
    a: np.ndarray, b: np.ndarray, jitter_scale: float = Defaults.jitter_scale
) -> np.ndarray[Any, np.dtype[np.unsignedinteger]]:
    """Calculate safe div function."""
    mask = b == 0
    b[mask] = 1
    inv_b = 1.0 / b
    result = a * inv_b

    if mask.sum() > 0:  # noqa
        result[mask.repeat(2, 2)] = np.random.randn(mask.sum() * 2) * jitter_scale  # noqa  # noqa

    return result

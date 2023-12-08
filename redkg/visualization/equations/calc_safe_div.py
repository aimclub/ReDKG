"""Calculate safe div module."""

import numpy as np

from redkg.visualization.config.parameters.defaults import Defaults


def safe_div(
        a: np.ndarray,
        b: np.ndarray,
        jitter_scale: float = Defaults.jitter_scale
):
    """Calculate safe div function."""
    mask = b == 0
    b[mask] = 1
    inv_b = 1.0 / b
    result = a * inv_b

    if mask.sum() > 0:  # noqa
        result[mask.repeat(2, 2)] = np.random.randn(  # noqa
            mask.sum() * 2  # noqa
        ) * jitter_scale

    return result

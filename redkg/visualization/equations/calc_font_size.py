"""Calculate font size module."""

import numpy as np

from redkg.visualization.config.parameters.defaults import Defaults
from redkg.visualization.utils.cached import cached


@cached()
def calculate_font_size(vertex_num):
    """Calculate font size function."""
    return Defaults.font_size_multiplier * np.exp(-vertex_num / Defaults.font_size_divider)

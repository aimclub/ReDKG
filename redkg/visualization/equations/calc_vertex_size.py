"""Calculate vertex size module."""

import numpy as np

from redkg.visualization.config.parameters.defaults import Defaults
from redkg.visualization.utils.cached import cached


@cached()
def calculate_vertex_size(vertex_num):
    """Calculate vertex size function."""
    return (
        Defaults.calculate_vertex_size_multiplier
        / np.sqrt(vertex_num + Defaults.calculate_vertex_size_divider)
        * Defaults.calculate_vertex_size_modifier
    )

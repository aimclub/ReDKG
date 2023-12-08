"""Calculate edge line width module."""

import numpy as np

from redkg.visualization.config.parameters.defaults import Defaults
from redkg.visualization.utils.cached import cached


@cached()
def calculate_edge_line_width(edge_list_length: int):
    """Calculate edge line width function."""
    return Defaults.edge_line_width_multiplier * np.exp(
        -edge_list_length / Defaults.edge_line_width_divider
    )

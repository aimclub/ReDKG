import numpy as np

from redkg.visualization.config.parameters.defaults import Defaults
from redkg.visualization.utils.cached import cached


@cached()
def calculate_vertex_line_width(vertex_num):
    return Defaults.vertex_line_width_multiplier * np.exp(-vertex_num / Defaults.vertex_line_width_divider)

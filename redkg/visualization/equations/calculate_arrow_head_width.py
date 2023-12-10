"""Calculate arrows head parameters module."""

from typing import Union

from redkg.visualization.config.parameters.defaults import Defaults
from redkg.visualization.config.types import TEdgeList


def calculate_arrow_head_width(
    edge_line_width: list, show_arrow: bool, edge_list: Union[TEdgeList, list[tuple[int, int]]]
):
    """Calculate arrows head parameters function."""
    return [Defaults.arrow_multiplier * w for w in edge_line_width] if show_arrow else [0] * len(edge_list)

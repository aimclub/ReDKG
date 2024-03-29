"""DrawEdgesContract module."""

from dataclasses import dataclass
from typing import Union

from redkg.visualization.config.types import TVectorCoordinates


@dataclass
class DrawEdgesContract:
    """Draw edges contract base class."""

    vertex_coordinates: TVectorCoordinates
    vertex_size: list
    edge_list: Union[list[tuple], list[list[int]]]
    edge_color: list
    edge_fill_color: list
    edge_line_width: list

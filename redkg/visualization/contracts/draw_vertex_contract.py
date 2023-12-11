"""DrawVertexContract module."""

from dataclasses import dataclass
from typing import Iterable, Union

from redkg.visualization.config.types import TVectorCoordinates


@dataclass
class DrawVertexContract:
    """Draw vertex contract base class."""

    vertex_coordinates: TVectorCoordinates
    vertex_label: Union[list[str], Iterable[str]]
    font_size: int
    font_family: str
    vertex_size: Union[list, Iterable[str]]
    vertex_color: list
    vertex_line_width: list

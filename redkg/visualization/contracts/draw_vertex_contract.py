"""DrawVertexContract module."""

from dataclasses import dataclass

from redkg.visualization.config.types import TVectorCoordinates


@dataclass
class DrawVertexContract:
    """DrawVertexContract base class."""
    vertex_coordinates: TVectorCoordinates
    vertex_label: list[str] | None
    font_size: int
    font_family: str
    vertex_size: list
    vertex_color: list
    vertex_line_width: list

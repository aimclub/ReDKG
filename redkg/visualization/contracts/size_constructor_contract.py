"""SizeConstructorContract module."""

from dataclasses import dataclass
from typing import Sized

from redkg.visualization.config.parameters.defaults import Defaults
from redkg.visualization.config.types import TEdgeList


@dataclass
class SizeConstructorContract:
    """Size сonstructor сontract base class."""

    vertex_num: int
    edge_list: TEdgeList | list | Sized = []
    vertex_size: float | list = Defaults.vertex_size
    vertex_line_width: float | list = Defaults.vertex_line_width
    edge_line_width: float | list = Defaults.edge_line_width
    font_size: float = Defaults.font_size

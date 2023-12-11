"""SizeConstructorContract module."""

from dataclasses import dataclass, field
from typing import Sized, Union

from redkg.visualization.config.parameters.defaults import Defaults
from redkg.visualization.config.types import TEdgeList


@dataclass
class SizeConstructorContract:
    """Size сonstructor сontract base class."""

    vertex_num: int
    edge_list: Union[TEdgeList, list, Sized] = field(default_factory=list)
    vertex_size: Union[float, list] = Defaults.vertex_size
    vertex_line_width: Union[float, list] = Defaults.vertex_line_width
    edge_line_width: Union[float, list] = Defaults.edge_line_width
    font_size: float = Defaults.font_size

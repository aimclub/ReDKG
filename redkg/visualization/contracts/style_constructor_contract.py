"""GraphStyleConstructorContract module."""

from dataclasses import dataclass
from typing import Union

from redkg.visualization.config.parameters.defaults import Defaults


@dataclass
class GraphStyleConstructorContract:
    """GraphStyle сonstructor сontract base class."""

    vertex_num: int
    edges_num: int
    vertex_color: Union[str, list] = Defaults.vertex_color
    edge_color: Union[str, list] = Defaults.edge_color
    edge_fill_color: Union[str, list] = Defaults.edge_fill_color

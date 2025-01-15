"""BaseVisualizationContract module."""

from dataclasses import dataclass, field
from typing import Any, Iterable, Union

from redkg.visualization.config.parameters.defaults import Defaults


@dataclass
class BaseVisualizationContract:
    """Base visualization contract base class."""

    edge_style: str = Defaults.edge_style
    edge_color: Union[str, list] = Defaults.edge_color
    edge_fill_color: Union[str, list[Any]] = Defaults.edge_fill_color
    edge_line_width: Union[float, list[Any]] = Defaults.edge_line_width
    vertex_label: Union[list[str], Iterable[str]] = field(default_factory=list)
    vertex_size: Union[float, list] = Defaults.vertex_size
    vertex_color: Union[str, list] = Defaults.vertex_color
    vertex_line_width: Union[float, list] = Defaults.vertex_line_width
    font_size: float = Defaults.font_size
    font_family: str = Defaults.font_family
    push_vertex_strength: float = Defaults.push_vertex_strength_vis
    push_edge_strength: float = Defaults.push_edge_strength_vis
    pull_edge_strength: float = Defaults.pull_edge_strength_vis
    pull_center_strength: float = Defaults.pull_center_strength_vis

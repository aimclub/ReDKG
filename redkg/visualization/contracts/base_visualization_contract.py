"""BaseVisualizationContract module."""

from dataclasses import dataclass

from redkg.visualization.config.parameters.defaults import Defaults


@dataclass
class BaseVisualizationContract:
    """Base visualization contract base class."""

    edge_style: str = Defaults.edge_style
    edge_color: str | list = Defaults.edge_color
    edge_fill_color: str | float | list = Defaults.edge_fill_color
    edge_line_width: str | float | list = Defaults.edge_line_width
    vertex_label: list | None = None
    vertex_size: float | list = Defaults.vertex_size
    vertex_color: str | list = Defaults.vertex_color
    vertex_line_width: float | list = Defaults.vertex_line_width
    font_size: float = Defaults.font_size
    font_family: str = Defaults.font_family
    push_vertex_strength: float = Defaults.push_vertex_strength_vis
    push_edge_strength: float = Defaults.push_edge_strength_vis
    pull_edge_strength: float = Defaults.pull_edge_strength_vis
    pull_center_strength: float = Defaults.pull_center_strength_vis

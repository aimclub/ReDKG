"""BaseGraphContract module."""

from dataclasses import dataclass

from redkg.visualization.config.types import TGraphEdgeList


@dataclass
class BaseGraphContract:
    """Base graph contract class."""

    vertex_num: int
    edge_list: TGraphEdgeList | None = None
    edge_weights: float | list[float] | None = None

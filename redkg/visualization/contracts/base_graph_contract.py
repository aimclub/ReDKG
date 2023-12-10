"""BaseGraphContract module."""

from dataclasses import dataclass
from typing import Any, Optional, Union

from redkg.visualization.config.types import TGraphEdgeList


@dataclass
class BaseGraphContract:
    """Base graph contract class."""

    vertex_num: int
    edge_list: tuple[Any, list[float]]
    edge_weights: Optional[Union[float, list[float]]]

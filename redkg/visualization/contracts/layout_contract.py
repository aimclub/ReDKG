"""LayoutContract module."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LayoutContract:
    """Layout сontract base class."""

    vertex_num: int
    edge_list: list[tuple[Any, ...]]
    push_vertex_strength: Optional[float]
    push_edge_strength: Optional[float]
    pull_edge_strength: Optional[float]
    pull_center_strength: Optional[float]

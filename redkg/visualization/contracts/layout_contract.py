"""LayoutContract module."""

from dataclasses import dataclass
from typing import Any


@dataclass
class LayoutContract:
    """Layout сontract base class."""

    vertex_num: int
    edge_list: list[tuple[Any, ...]]
    push_vertex_strength: float | None
    push_edge_strength: float | None
    pull_edge_strength: float | None
    pull_center_strength: float | None

"""BaseGraphContract module."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class BaseGraphContract:
    """Base graph contract class."""

    vertex_num: int
    edge_list: tuple[Any, list[float]]
    edge_weights: Optional[Union[float, list[float]]] = field(default_factory=list)

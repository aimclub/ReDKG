"""HypergraphContract module."""

from dataclasses import dataclass
from typing import Optional

from redkg.visualization.contracts.base_graph_contract import BaseGraphContract


@dataclass
class HypergraphContract(BaseGraphContract):
    """Hypergraph contract base class."""

    vertex_weight: Optional[list[float]] = None
    edge_num: int = 0

"""GraphContract module."""

from dataclasses import dataclass

from redkg.visualization.contracts.base_graph_contract import BaseGraphContract


@dataclass
class GraphContract(BaseGraphContract):
    """Graph contract base class."""

    edge_num: int = 0

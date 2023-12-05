from dataclasses import dataclass

from redkg.visualization.config.parameters.edge_styles import EdgeStyles
from redkg.visualization.contracts.base_visualization_contract import BaseVisualizationContract
from redkg.visualization.contracts.hypergraph_contract import HypergraphContract


@dataclass
class HypergraphVisualizationContract(BaseVisualizationContract):
    graph: HypergraphContract = None
    edge_style: str = EdgeStyles.circle


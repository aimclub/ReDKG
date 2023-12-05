from dataclasses import dataclass

from redkg.visualization.contracts.base_visualization_contract import BaseVisualizationContract
from redkg.visualization.contracts.graph_contract import GraphContract


@dataclass
class GraphVisualizationContract(BaseVisualizationContract):
    graph: GraphContract = None

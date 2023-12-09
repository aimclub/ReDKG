"""Draw simple graph module."""

from torch import tensor

from redkg.visualization.contracts.graph_contract import GraphContract
from redkg.visualization.contracts.graph_visualization_contract import (
    GraphVisualizationContract
)
from redkg.visualization.graph_visualization import GraphVisualizer
from visualization.mock_data.mock_data import SIMPLE_EDGE_LIST

graph_contract: GraphContract = GraphContract(
    vertex_num=10,
    edge_list=(  # noqa
        SIMPLE_EDGE_LIST,
        [1.0] * 12,
    ),
    edge_num=12,
    edge_weights=tensor([1.0] * 24),  # noqa
)

vis_contract: GraphVisualizationContract = GraphVisualizationContract(
    graph=graph_contract
)

vis: GraphVisualizer = GraphVisualizer(vis_contract)
fig = vis.draw()
fig.show()

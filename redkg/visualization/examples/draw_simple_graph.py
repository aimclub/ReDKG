"""Draw simple graph module."""

from torch import tensor

from redkg.visualization.contracts.graph_contract import GraphContract
from redkg.visualization.contracts.graph_visualization_contract import (
    GraphVisualizationContract
)
from redkg.visualization.graph_visualization import GraphVisualizer

graph_contract: GraphContract = GraphContract(
    vertex_num=10,
    edge_list=(  # noqa
        [
            (0, 7), (2, 7), (4, 9), (3, 7), (1, 8), (5, 7), (2, 3), (4, 5),
            (5, 6), (4, 8), (6, 9), (4, 7)
        ],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ),
    edge_num=12,
    edge_weights=tensor(  # noqa
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    ),
)

vis_contract: GraphVisualizationContract = GraphVisualizationContract(
    graph=graph_contract
)

vis: GraphVisualizer = GraphVisualizer(vis_contract)
fig = vis.draw()
fig.show()

"""Draw simple hypergraph module."""

from torch import tensor

from redkg.visualization.contracts.hypergraph_contract import HypergraphContract
from redkg.visualization.contracts.hypergraph_visualization_contract import HypergraphVisualizationContract
from redkg.visualization.hypergraph_visualization import HypergraphVisualizer
from redkg.visualization.mock_data.mock_data import SIMPLE_HEDGE_LIST

graph_contract: HypergraphContract = HypergraphContract(
    vertex_num=10,
    edge_list=(  # noqa
        SIMPLE_HEDGE_LIST,
        [1.0] * 8,
    ),
    edge_num=8,
    edge_weights=tensor([1.0] * 10),  # noqa
)

vis_contract: HypergraphVisualizationContract = (
    HypergraphVisualizationContract(
        graph=graph_contract
    )
)

vis: HypergraphVisualizer = HypergraphVisualizer(vis_contract)
fig = vis.draw()
fig.show()

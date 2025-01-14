"""Module containing Bellman-Ford layer for node classification GNN example."""

import torch
import torch.nn as nn
from raw_bellman_ford.layers.bellman_ford_orig import BellmanFordLayer


class GNNWithBellmanFord(nn.Module):
    """
    Graph Neural Network (GNN) with Bellman-Ford layer for node classification.

    Parameters:
    - num_nodes: Number of nodes in the graph.
    - num_features: Dimensionality of node features.
    - num_classes: Number of classes for node classification.
    """

    def __init__(self, num_nodes, num_features, num_classes):
        """
        Initialize the GNNWithBellmanFord model.

        Parameters:
        - num_nodes: Number of nodes in the graph.
        - num_features: Dimensionality of node features.
        - num_classes: Number of classes for node classification.
        """
        super(GNNWithBellmanFord, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes

        self.bellman_ford_layer = BellmanFordLayer(num_nodes)
        self.node_embedding = nn.Embedding(num_nodes, num_features)
        self.fc = nn.Linear(num_features + num_nodes, num_classes)

    def forward(self, adj_matrix, source_node):
        """
        Forward pass of the GNNWithBellmanFord model.

        Parameters:
        - adj_matrix: Adjacency matrix of the graph.
        - source_node: Source node for the Bellman-Ford algorithm.

        Returns:
        - output: GNN output after the Bellman-Ford layer and fully connected layer.
        - has_negative_cycle: Boolean indicating whether the graph contains a negative weight cycle.
        """
        distances, predecessors, has_negative_cycle = self.bellman_ford_layer(adj_matrix, source_node)

        node_features = self.node_embedding(torch.arange(self.num_nodes))
        node_features = torch.cat([node_features, distances], dim=1)

        output = self.fc(node_features)

        return output, has_negative_cycle


if __name__ == "__main__":
    # Example 1
    num_nodes_1 = 4
    adj_matrix_1 = torch.tensor(
        [
            [0, 2, float("inf"), 1],
            [float("inf"), 0, -1, float("inf")],
            [float("inf"), float("inf"), 0, -2],
            [float("inf"), float("inf"), float("inf"), 0],
        ]
    )
    source_node_1 = 0

    gnn_model_1 = GNNWithBellmanFord(num_nodes_1, num_features=5, num_classes=2)
    output_1, has_negative_cycle_1 = gnn_model_1(adj_matrix_1, source_node_1)

    if has_negative_cycle_1:
        print("Example 1: The graph contains a negative weight cycle")
    else:
        print("Example 1: GNN output:", output_1)

    # Example 2
    num_nodes_2 = 5
    adj_matrix_2 = torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]])
    source_node_2 = 2

    gnn_model_2 = GNNWithBellmanFord(num_nodes_2, num_features=4, num_classes=3)
    output_2, has_negative_cycle_2 = gnn_model_2(adj_matrix_2, source_node_2)

    if has_negative_cycle_2:
        print("Example 2: The graph contains a negative weight cycle")
    else:
        print("Example 2: GNN output:", output_2)

    # Example 3
    num_nodes_3 = 3
    adj_matrix_3 = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    source_node_3 = 0

    gnn_model_3 = GNNWithBellmanFord(num_nodes_3, num_features=4, num_classes=2)
    output_3, has_negative_cycle_3 = gnn_model_3(adj_matrix_3, source_node_3)

    if has_negative_cycle_3:
        print("Example 3: The graph contains a negative weight cycle")
    else:
        print("Example 3: GNN output:", output_3)

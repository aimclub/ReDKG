"""Module containing BellmanFordLayer testing"""

import torch
from raw_bellman_ford.layers.bellman_ford_orig import BellmanFordLayer

def test_bellman_ford_no_negative_cycle():
    """
    Test the Bellman-Ford algorithm implementation when there is no negative cycle.

    This function creates a test case for the Bellman-Ford algorithm. It sets up the necessary inputs,
    including the number of nodes, the source node, and the adjacency matrix. Then, it calls the BellmanFordLayer
    class to perform the Bellman-Ford algorithm on the given inputs.

    Parameters:
        None

    Returns:
        None. This function only performs assertions to check the correctness of the algorithm output.
    """
    num_nodes = 4
    source_node = 0

    adj_matrix = torch.tensor([[0, 2, 1, float('inf')],
                               [float('inf'), 0, -1, float('inf')],
                               [float('inf'), float('inf'), 0, -2],
                               [float('inf'), float('inf'), float('inf'), 0]])

    bellman_ford_layer = BellmanFordLayer(num_nodes)

    distances, predecessors, has_negative_cycle = bellman_ford_layer(adj_matrix, source_node)

    assert not has_negative_cycle

def test_bellman_ford_with_negative_cycle():
    """
    Run a test case for the Bellman-Ford algorithm with a negative cycle.

    Parameters:
        None

    Returns:
        None
    """
    num_nodes = 4
    source_node = 0

    adj_matrix = torch.tensor([[0, 2, 1, float('inf')],
                               [float('inf'), 0, -1, float('inf')],
                               [3, float('inf'), 0, -2],
                               [float('inf'), 1, float('inf'), 0]])

    bellman_ford_layer = BellmanFordLayer(num_nodes)

    distances, predecessors, has_negative_cycle = bellman_ford_layer(adj_matrix, source_node)

    assert has_negative_cycle

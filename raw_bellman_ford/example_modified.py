import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from bellman_ford_modified import BellmanFordLayerModified

def visualize_graph(adj_matrix, node_labels=None):
    # Проверяем наличие рёбер
    if torch.max(adj_matrix) == float('inf'):
        print("Graph has no edges to visualize.")
        return

    G = nx.from_numpy_array(np.array(adj_matrix), create_using=nx.DiGraph())

    pos = nx.spring_layout(G)  # Можете выбрать другой метод размещения вершин

    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black", font_weight="bold", arrowsize=10)

    if node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="red", font_weight="bold")

    plt.show()


def run_and_visualize_example(num_nodes, adj_matrix, source_node, num_features=5):
    bellman_ford_layer = BellmanFordLayerModified(num_nodes, num_features)
    node_features, diameter, eccentricity, radius, central_nodes, peripheral_nodes, closeness_centrality, degree_centrality, betweenness_centrality = bellman_ford_layer(adj_matrix, source_node)

    print("Node Features:")
    print(node_features)
    print("Graph Diameter:", diameter)
    print("Graph Eccentricity:", eccentricity)
    print("Graph Radius:", radius)
    print("Central Nodes:", central_nodes)
    print("Peripheral Nodes:", peripheral_nodes)
    print("Closeness Centrality:", closeness_centrality)
    print("Degree Centrality:", degree_centrality)
    print("Betweenness Centrality:", betweenness_centrality)

    # Visualize the graph
    visualize_graph(adj_matrix, node_labels={i: f'{i}\n{node_features[i][-1]:.2f}' for i in range(num_nodes)})

if __name__ == "__main__":
    # Example 1
    num_nodes_1 = 4
    adj_matrix_1 = torch.tensor([[0, 2, float('inf'), 1],
                                 [float('inf'), 0, -1, float('inf')],
                                 [float('inf'), float('inf'), 0, -2],
                                 [float('inf'), float('inf'), float('inf'), 0]])
    source_node_1 = 0

    run_and_visualize_example(num_nodes_1, adj_matrix_1, source_node_1)

    # Example 2
    num_nodes_2 = 4
    adj_matrix_2 = torch.tensor([[0, 2, 1, float('inf')],
                                 [float('inf'), 0, -1, float('inf')],
                                 [float('inf'), float('inf'), 0, -2],
                                 [float('inf'), float('inf'), float('inf'), 0]])
    source_node_2 = 0

    run_and_visualize_example(num_nodes_2, adj_matrix_2, source_node_2)

    # # Example 3
    # num_nodes_3 = 4
    # adj_matrix_3 = torch.tensor([[0, 2, 1, 3],
    #                              [-1, 0, -1, 4],
    #                              [5, 2, 0, -2],
    #                              [2, 3, 1, 0]])
    # source_node_3 = 0

    # run_and_visualize_example(num_nodes_3, adj_matrix_3, source_node_3)

    # Example 4
    num_nodes_4 = 4
    adj_matrix_4 = torch.tensor([[0, 2, 0, 1],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 2],
                                 [0, 0, 0, 0]])
    source_node_4 = 0

    run_and_visualize_example(num_nodes_4, adj_matrix_4, source_node_4)

    num_nodes_1 = 5
    adj_matrix_1 = torch.tensor([[0, 2, float('inf'), 1, 3],
                             [float('inf'), 0, -1, float('inf'), 2],
                             [float('inf'), float('inf'), 0, -2, 1],
                             [float('inf'), float('inf'), float('inf'), 0, 1],
                             [float('inf'), float('inf'), float('inf'), float('inf'), 0]])
    source_node_1 = 0
    run_and_visualize_example(num_nodes_1, adj_matrix_1, source_node_1)

    num_nodes_3 = 6
    adj_matrix_3 = torch.full((num_nodes_3, num_nodes_3), float('inf'))
    source_node_3 = 0
    run_and_visualize_example(num_nodes_3, adj_matrix_3, source_node_3)

    num_nodes_1 = 5
    adj_matrix_1 = torch.tensor([[0, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1],
                             [1, 0, 0, 0, 0]])
    source_node_1 = 0

    run_and_visualize_example(num_nodes_1, adj_matrix_1, source_node_1)

    num_nodes_3 = 5
    adj_matrix_3 = torch.tensor([[0, 2, 1, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 2],
                             [0, 0, 0, 0, 0]])
    source_node_3 = 0

    run_and_visualize_example(num_nodes_3, adj_matrix_3, source_node_3)





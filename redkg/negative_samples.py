import random


def common_neighbors(edge_index, num_nodes):
    """
    Create a dictionary of neighbors for each node.

    Args:
        edge_index (Tensor): Edge indices, shape `(2, num_edges)`.
        num_nodes (int): Total number of nodes.

    Returns:
        dict: A dictionary mapping each node to its set of neighbors.
    """
    # Создание списка соседей для каждого узла
    neighbors = {i: set() for i in range(num_nodes)}
    for edge in edge_index.t().tolist():
        neighbors[edge[0]].add(edge[1])
        neighbors[edge[1]].add(edge[0])

    return neighbors


def generate_negative_samples(edge_index, num_nodes, num_neg_samples, max_attempts=1000):
    """
    Generate negative samples for link prediction.

    Args:
        edge_index (Tensor): Edge indices, shape `(2, num_edges)`.
        num_nodes (int): Total number of nodes.
        num_neg_samples (int): Number of negative samples.
        max_attempts (int, optional): Max attempts to find samples (default 1000).

    Returns:
        list: Negative samples, each a pair of nodes `[node1, node2]`.

    Example:
        generate_negative_samples(torch.tensor([[0, 1], [1, 2]]), 3, 2)
        # Returns [[0, 2], [1, 0]]
    """
    neighbors = common_neighbors(edge_index, num_nodes)
    negative_samples = []
    attempts = 0

    while len(negative_samples) < num_neg_samples and attempts < max_attempts:
        node1 = random.choice(range(num_nodes))
        node2 = random.choice(range(num_nodes))

        # Проверяем, что узлы не связаны и имеют общих соседей
        if node1 != node2 and node2 not in neighbors[node1]:
            common_neigh = neighbors[node1].intersection(neighbors[node2])
            # Условие можно ослабить, уменьшив требуемое количество общих соседей
            if len(common_neigh) > 0:  # Узлы имеют общих соседей
                negative_samples.append([node1, node2])

        attempts += 1

    return negative_samples

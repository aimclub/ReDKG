import random


def common_neighbors(edge_index, num_nodes):
    # Создание списка соседей для каждого узла
    neighbors = {i: set() for i in range(num_nodes)}
    for edge in edge_index.t().tolist():
        neighbors[edge[0]].add(edge[1])
        neighbors[edge[1]].add(edge[0])

    return neighbors


def generate_negative_samples(edge_index, num_nodes, num_neg_samples, max_attempts=1000):
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

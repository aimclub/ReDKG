import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

class HBellmanFord:
    def __init__(self, nodes, edges, node_types, edge_types, edge_weights, hyperedges, hyperedge_types, hyperedge_weights):
        self.nodes = nodes
        self.edges = edges
        self.node_types = node_types
        self.edge_types = edge_types
        self.edge_weights = edge_weights
        self.hyperedges = hyperedges
        self.hyperedge_types = hyperedge_types
        self.hyperedge_weights = hyperedge_weights

    def initialize_distances(self, source):
        distances = {node: float('inf') for node in self.nodes}
        distances[source] = 0
        return distances

    def relax(self, u, v, weight, distances):
        if distances[u] + weight < distances[v]:
            distances[v] = distances[u] + weight

    def eccentricity(self, distances):
        return np.max([d for d in distances.flatten() if not np.isinf(d)])

    def radius(self, distances):
        flat_distances = distances.flatten()
        return np.min(flat_distances[~np.isinf(flat_distances)])

    def diameter(self, distances):
        flat_distances = distances.flatten()
        return np.max(flat_distances[~np.isinf(flat_distances)])

    def central_nodes(self, distances):
        radius = self.radius(distances)
        return [node for node, distance in enumerate(distances.flatten()) if np.isinf(distance) or distance == radius]

    def peripheral_nodes(self, distances):
        diameter = self.diameter(distances)
        return [node for node, distance in enumerate(distances.flatten()) if distance == diameter]

    def closeness_centrality(self, node, distances):
        total_shortest_paths = 0
        total_connected_nodes = 0
        for l in range(len(self.nodes)):
            if l != node:
                for j in range(len(self.nodes)):
                    if j != l and j != node and not np.isinf(distances[l][j]):
                        total_shortest_paths += distances[l][j]
                        total_connected_nodes += 1

        if total_connected_nodes == 0:
            return 0  # Изолированная вершина, возвращаем 0

        return total_connected_nodes / total_shortest_paths


    def degree_centrality(self, node):
        return self.node_types.get(node, 0) / (len(self.nodes) - 1)

    def betweenness_centrality(self, node, distances):
        total_shortest_paths = 0
        for l in range(len(self.nodes)):
            if l != node:
                for j in range(len(self.nodes)):
                    if j != l and j != node:
                        total_shortest_paths += distances[l][node] / distances[l][j]
        return total_shortest_paths

    def bellman_ford(self, node_criteria, edge_criteria, hyperedge_criteria):
        distance_matrix = []

        for source_node in self.nodes:
            distances = self.initialize_distances(source_node)

            for i in range(len(self.nodes) - 1):
                for edge in self.edges:
                    u, v = edge
                    if (node_criteria == 0) or (self.node_types[v] in node_criteria):
                        weight = self.edge_weights[edge]
                        self.relax(u, v, weight, distances)

                for hyperedge in self.hyperedges:
                    u, v, w = hyperedge
                    if (edge_criteria == 0) or (self.edge_types[(u, v)] in edge_criteria):
                        weight = self.hyperedge_weights[hyperedge]
                        self.relax(u, w, weight, distances)

            distance_matrix.append([distances[i] for i in self.nodes])

        return distance_matrix

    def visualize_hypergraph(self):
        G = nx.MultiDiGraph()

        for node in self.nodes:
            node_label = f'gv({node})/tp({self.node_types[node]})'
            G.add_node(node, color='skyblue', label=node_label)

        for edge in self.edges:
            u, v = edge
            edge_label = f'ge({edge})/tp({self.edge_types[edge]})/wt({self.edge_weights[edge]})'
            G.add_edge(u, v, color='black', label=edge_label)

        for hyperedge in self.hyperedges:
            hyperedge_label = f'ge({hyperedge})/tp({self.hyperedge_types[hyperedge]})/wt({self.hyperedge_weights[hyperedge]})'
            G.add_node(hyperedge, color='red', label=hyperedge_label)
            for u in hyperedge:
                G.add_edge(u, hyperedge, color='red', label='')

        pos = nx.spring_layout(G)

        edges = G.edges()
        colors = [G[u][v][0]['color'] if 'color' in G[u][v][0] else 'black' for u, v in edges]

        node_labels = {node: G.nodes[node]['label'] for node in G.nodes}
        nx.draw(G, pos, with_labels=True, font_weight='bold', edgelist=edges, edge_color=colors,
                node_color='skyblue', node_size=1000, font_size=8, labels=node_labels)

        edge_labels = {(u, v): G[u][v][0]['label'] if 'label' in G[u][v][0] else '' for (u, v) in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.show()

if __name__ == "__main__":
    nodes = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (2, 1), (3, 4), (3, 5)]
    hyperedges = [(1, 2, 3), (3, 5, 4)]
    node_types = {1: 0, 2: 1, 3: 0, 4: 0, 5: 0}
    edge_types = {(1, 2): 1, (2, 3): 2, (2, 1): 1,  (3, 4): 1, (3, 5): 1}
    edge_weights = {(1, 2): 5.4, (2, 3): 2.2,  (2, 1): 2.1,  (3, 4): 1, (3, 5): 1}
    hyperedge_types = {(1, 2, 3): 1, (3, 5, 4): 1}
    hyperedge_weights = {(1, 2, 3): 3, (3, 5, 4): 2}

    bellman_ford = HBellmanFord(nodes, edges, node_types, edge_types, edge_weights, hyperedges, hyperedge_types, hyperedge_weights)
    node_criteria_all_types = 0
    node_criteria_specific_types = [0, 1]
    edge_criteria_all_types = 0
    edge_criteria_specific_types = [1]
    hyperedge_criteria_vertices = 0
    hyperedge_criteria_hyperedges = [1]

    distance_matrix = bellman_ford.bellman_ford(node_criteria_all_types, edge_criteria_all_types, hyperedge_criteria_vertices)

    bellman_ford.visualize_hypergraph()

    print("Full Distance Matrix:")
    for row in distance_matrix:
        print(row)

    distances = np.array(distance_matrix)
    print("Eccentricity:", bellman_ford.eccentricity(distances))
    print("Radius:", bellman_ford.radius(distances))
    print("Diameter:", bellman_ford.diameter(distances))
    print("Central Nodes:", bellman_ford.central_nodes(distances))
    print("Peripheral Nodes:", bellman_ford.peripheral_nodes(distances))

    for node in range(len(nodes)):
        print(f"\nNode {node} Centrality:")
        print("Closeness Centrality:", bellman_ford.closeness_centrality(node, distances))
        print("Degree Centrality:", bellman_ford.degree_centrality(node))
        print("Betweenness Centrality:", bellman_ford.betweenness_centrality(node, distances))


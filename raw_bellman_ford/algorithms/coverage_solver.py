"""Module containing class HypergraphCoverageSolver."""

import numpy as np

class HypergraphCoverageSolver:
    def __init__(self, nodes, edges, hyperedges, node_types, edge_weights, hyperedge_weights, hyperedge_types):
        """
        Initializes the HypergraphCoverageSolver object.

        Parameters:
        - nodes: List of hypergraph nodes.
        - edges: List of ordinary hypergraph edges.
        - hyperedges: List of hyper-hyperedges.
        - node_types: Dictionary with node types.
        - edge_weights: Dictionary with weights of ordinary edges.
        - hyperedge_weights: Dictionary with weights of hyper-hyperedges.
        - hyperedge_types: Dictionary with types of hyper-hyperedges.
        """
        self.nodes = nodes
        self.edges = edges
        self.hyperedges = hyperedges
        self.node_types = node_types
        self.edge_weights = edge_weights
        self.hyperedge_weights = hyperedge_weights
        self.hyperedge_types = hyperedge_types

    def get_edge_weight(self, edge):
        """
        Gets the weight of an ordinary edge.

        Parameters:
        - edge: Edge.

        Returns:
        - Edge weight.
        """
        return self.edge_weights.get(edge, 0)

    def get_hyperedge_weight(self, hyperedge):
        """
        Gets the weight of a hyper-hyperedge.

        Parameters:
        - hyperedge: Hyper-hyperedge.

        Returns:
        - Hyper-hyperedge weight.
        """
        return self.hyperedge_weights.get(hyperedge, 0)

    def compute_shortest_distances(self):
        """
        Computes the matrix of shortest distances.

        Returns:
        - Matrix of shortest distances.
        """
        num_nodes = len(self.nodes)
        num_hyperedges = len(self.hyperedges)

        hyperedge_index = {hyperedge: num_nodes + i for i, hyperedge in enumerate(self.hyperedges)}

        distance_matrix = np.zeros((num_nodes + num_hyperedges, num_nodes + num_hyperedges))
        distance_matrix.fill(float('inf'))
        np.fill_diagonal(distance_matrix, 0)

        for edge in self.edges:
            u, v = edge
            if isinstance(u, tuple):
                u = hyperedge_index[u]
            if isinstance(v, tuple):
                v = hyperedge_index[v]
            distance_matrix[u-1, v-1] = self.get_edge_weight(edge)
            distance_matrix[v-1, u-1] = self.get_edge_weight(edge)

        for hyperedge in self.hyperedges:
            hyperedge_index_value = hyperedge_index[hyperedge]
            hyperedge_weight = self.get_hyperedge_weight(hyperedge)

            for node in hyperedge:
                node_index = node - 1
                if hyperedge_index_value != node_index:
                    distance_matrix[node_index, hyperedge_index_value] = hyperedge_weight
                    distance_matrix[hyperedge_index_value, node_index] = hyperedge_weight

            if hyperedge_index_value < len(self.nodes):
                distance_matrix[hyperedge_index_value, hyperedge_index_value] = 0

        return distance_matrix

    def print_distance_matrix(self):
        """
        Prints the matrix of shortest distances.
        """
        print("Full Distance Matrix:")
        for i, row in enumerate(self.compute_shortest_distances()):
            node_or_hyperedge = f"gv{i + 1}" if i < len(self.nodes) else f"he{i + 1 - len(self.nodes)}"
            print(f"{node_or_hyperedge}: {row}")

    def can_cover_with_drone(self, drone_radius):
        """
        Checks the possibility of covering objects using a drone.

        Parameters:
        - drone_radius: Drone operating radius.

        Returns:
        - True if covering is possible, False otherwise.
        """
        distance_matrix = self.compute_shortest_distances()
        for i, node in enumerate(self.nodes):
            if self.node_types[node] == 1:  # Hyper-hyperedge
                diameter_within_hyperedge = np.max(distance_matrix[i, :len(self.nodes)])
                if diameter_within_hyperedge > drone_radius:
                    return False
            elif distance_matrix[i, i] > drone_radius:
                return False
        return True


class HypergraphMetricsCalculator:
    def __init__(self, distance_matrix):
        """
        Initializes the HypergraphMetricsCalculator object.

        Parameters:
        - distance_matrix: Distance matrix.
        """
        self.distance_matrix = distance_matrix

    def eccentricity(self, node):
        """
        Calculates the eccentricity of a node.

        Parameters:
        - node: Node.

        Returns:
        - Node eccentricity.
        """
        distances = self.distance_matrix[node - 1]
        finite_distances = distances[np.isfinite(distances)]
        return np.max(finite_distances) if len(finite_distances) > 0 else 0

    def radius(self):
        """
        Calculates the radius of the hypergraph.

        Returns:
        - Hypergraph radius.
        """
        max_distances = np.max(self.distance_matrix, axis=1)
        finite_distances = max_distances[np.isfinite(max_distances)]
        return np.min(finite_distances) if len(finite_distances) > 0 else 0

    def diameter(self):
        """
        Calculates the diameter of the hypergraph.

        Returns:
        - Hypergraph diameter.
        """
        max_distance = np.max(self.distance_matrix)
        return max_distance if np.isfinite(max_distance) else 0

    def central_nodes(self):
        """
        Calculates central nodes of the hypergraph.

        Returns:
        - List of central nodes.
        """
        min_eccentricity = self.radius()
        central_nodes = [i + 1 for i, eccentricity in enumerate(np.max(self.distance_matrix, axis=1)) if eccentricity == min_eccentricity]
        return central_nodes

    def peripheral_nodes(self):
        """
        Calculates peripheral nodes of the hypergraph.

        Returns:
        - List of peripheral nodes.
        """
        max_eccentricity = np.max(np.max(self.distance_matrix, axis=1))
        peripheral_nodes = [i + 1 for i, eccentricity in enumerate(np.max(self.distance_matrix, axis=1)) if eccentricity == max_eccentricity]
        return peripheral_nodes

    def closeness_centrality(self, node):
        """
        Calculates closeness centrality of a node.

        Parameters:
        - node: Node.

        Returns:
        - Closeness centrality value.
        """
        inverse_distances = 1 / self.distance_matrix[node - 1]
        inverse_distances[np.isinf(inverse_distances)] = 0
        sum_inverse_distances = np.sum(inverse_distances)
        closeness_centrality = 0 if sum_inverse_distances == 0 else 1 / sum_inverse_distances
        return closeness_centrality if not np.isinf(closeness_centrality) else 0

    def betweenness_centrality(self, node):
        """
        Calculates betweenness centrality of a node.

        Parameters:
        - node: Node.

        Returns:
        - Betweenness centrality value.
        """
        betweenness_values = np.zeros_like(self.distance_matrix)
        betweenness_values[np.isinf(self.distance_matrix)] = 0

        for k in range(len(self.distance_matrix)):
            for i in range(len(self.distance_matrix)):
                for j in range(len(self.distance_matrix)):
                    if i != j and i != k and j != k and self.distance_matrix[i, j] != np.inf and self.distance_matrix[i, k] != 0 and self.distance_matrix[k, j] != 0:
                        betweenness_values[i, j] += (self.distance_matrix[i, k] + self.distance_matrix[k, j]) / self.distance_matrix[i, j]

        betweenness_centrality = np.sum(betweenness_values) / 2
        return betweenness_centrality if not np.isinf(betweenness_centrality) else 0

    def degree_centrality(self, node):
        """
        Calculates degree centrality of a node.

        Parameters:
        - node: Node.

        Returns:
        - Degree centrality value.
        """
        return np.sum(self.distance_matrix[node - 1] != np.inf) / (len(self.distance_matrix) - 1)
    
if __name__ == "__main__":
    nodes = [1, 2, 3, 4, 5, 6]
    edges = [(1, 2), (2, 3),(1, 4), (3, 4), ((1, 2, 3, 4), 5), ((1, 2, 3, 4), 6), (5, 6)]
    hyperedges = [(1, 2, 3, 4)]
    node_types = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    edge_weights = {(1, 2): 0.6, (2, 3): 0.5,  (3, 4): 0.6, (1, 4): 0.5, ((1, 2, 3, 4), 5): 0.2, ((1, 2, 3, 4), 6): 0.8, (5, 6): 0.7}
    hyperedge_weights = {(1, 2, 3, 4): 0.6}
    hyperedge_types = {(1, 2, 3, 4): 1}

    hypergraph_solver = HypergraphCoverageSolver(nodes, edges, hyperedges, node_types, edge_weights, hyperedge_weights, hyperedge_types)
    hypergraph_solver.print_distance_matrix()

    distance_matrix = hypergraph_solver.compute_shortest_distances()
    metrics_calculator = HypergraphMetricsCalculator(distance_matrix)

    # Node metrics calculation
    for node in nodes:
        print(f"Node {node}:")
        print(f"Eccentricity: {metrics_calculator.eccentricity(node)}")
        print(f"Closeness Centrality: {metrics_calculator.closeness_centrality(node)}")
        print(f"Degree Centrality: {metrics_calculator.degree_centrality(node)}")
        print(f"Betweenness Centrality: {metrics_calculator.betweenness_centrality(node)}")
        print("---")

    # Graph metrics calculation
    print(f"Radius: {metrics_calculator.radius()}")
    print(f"Diameter: {metrics_calculator.diameter()}")
    print(f"Central Nodes: {metrics_calculator.central_nodes()}")
    print(f"Peripheral Nodes: {metrics_calculator.peripheral_nodes()}")

    drone_radius = 1.0  # Drone radius
    coverage_solver = HypergraphCoverageSolver(nodes, edges, hyperedges, node_types, edge_weights, hyperedge_weights, hyperedge_types)
    can_cover = coverage_solver.can_cover_with_drone(drone_radius)

    if can_cover:
        print("The drone can cover the hypergraph")
    else:
        print("It's not possible for the drone to cover the hypergraph")

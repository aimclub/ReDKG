import numpy as np

class HypergraphCoverageSolver:
    def __init__(self, nodes, edges, hyperedges, node_types, edge_types, hyperedge_types):
        self.nodes = nodes
        self.edges = edges
        self.hyperedges = hyperedges
        self.node_types = node_types
        self.edge_types = edge_types
        self.hyperedge_types = hyperedge_types

    def can_cover_objects(self, drone_radius):
        min_radius = self.calculate_min_radius()

        return min_radius <= drone_radius

    def calculate_min_radius(self):
        min_radius = 0.0

        for edge in self.edges:
            min_radius = max(min_radius, self.get_edge_weight(edge))

        for hyperedge in self.hyperedges:
            min_radius = max(min_radius, self.get_hyperedge_weight(hyperedge))

        return min_radius

    def get_edge_weight(self, edge):
        return self.edge_types.get(edge, 1)

    def get_hyperedge_weight(self, hyperedge):
        return self.hyperedge_types.get(hyperedge, 1)

if __name__ == "__main__":
    nodes = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (3, 1), ((1, 2, 3), 4), ((1, 2, 3), 5), (4, 5)]
    hyperedges = [(1, 2, 3)]
    node_types = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    edge_types = {(1, 2): 1.4, (2, 3): 1.5, (3, 1): 1.6, ((1, 2, 3), 4): 2.5, ((1, 2, 3), 5): 24.6, (4, 5): 25.7}
    hyperedge_types = {(1, 2, 3): 1}

    hypergraph_solver = HypergraphCoverageSolver(nodes, edges, hyperedges, node_types, edge_types, hyperedge_types)

    drone_radius = 40

    if hypergraph_solver.can_cover_objects(drone_radius):
        print("БПЛА может покрыть все объекты в гиперграфе.")
    else:
        print("БПЛА не может покрыть все объекты в гиперграфе.")

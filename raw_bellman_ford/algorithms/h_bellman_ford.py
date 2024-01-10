"""Module containing HBellmanFord and HypergraphMetrics classes."""

import numpy as np
import pandas as pd
import networkx as nx

class HBellmanFord:
    def __init__(self, nodes, edges, hyperedges, criteria = None):
        """
        Initializes an instance of the class.

        Args:
            nodes (dict): A dictionary representing the nodes.
            edges (list): A list representing the edges.
            hyperedges (dict): A dictionary representing the hyperedges.
            node_types (dict, optional): A dictionary representing the node types. Defaults to None.
            edge_weights (dict, optional): A dictionary representing the edge weights. Defaults to None.
            hyperedge_weights (dict, optional): A dictionary representing the hyperedge weights. Defaults to None.
            hyperedge_types (dict, optional): A dictionary representing the hyperedge types. Defaults to None.
        """
        self.nodes = list(nodes.keys())
        self.edges = edges
        self.hyperedges = list(hyperedges.keys())
        self.edge_weights = {edge['nodes']: {"weight": edge['weight'], "attributes": edge['attributes']}  for edge in self.edges} if self.edges else {}
        self.hyperedge_weights = {key: value.setdefault("weight", np.inf) for key, value in hyperedges.items()} if hyperedges else {}
        self.criteria = criteria if criteria else {}

    def to_dict(self):
        """
        Convert the graph object to a dictionary representation.

        Returns:
            dict: A dictionary containing the graph data.
                - "nodes" (list): A list of nodes in the graph.
                - "edges" (list): A list of edges in the graph.
                - "hyperedges" (list): A list of hyperedges in the graph.
                - "node_types" (list): A list of node types in the graph.
                - "edge_weights" (dict): A dictionary mapping edges to their weights.
                - "hyperedge_weights" (dict): A dictionary mapping hyperedges to their weights.
                - "hyperedge_types" (list): A list of hyperedge types in the graph.
        """
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "hyperedges": self.hyperedges,
            "node_types": self.node_types,
            "edge_weights": self.edge_weights,
            "hyperedge_weights": self.hyperedge_weights,
            "hyperedge_types": self.hyperedge_types,
        }

class HypergraphMetrics:
    def __init__(self, hypergraph: HBellmanFord):
        """
        Initializes an instance of the class.

        Parameters:
            hypergraph (Hypergraph): The hypergraph object to be used.

        Returns:
            None
        """
        self.hypergraph = hypergraph
        self.matrix = self.create_matrix()
        self.graph = self.create_graph()
        self.num_nodes = len(self.matrix)
    
    def create_matrix(self):
        """
        Generates the adjacency matrix of the hypergraph.

        Returns:
            numpy.ndarray: The adjacency matrix of the hypergraph.
        """
        num_nodes = len(self.hypergraph.nodes) + len(self.hypergraph.hyperedges)
        matrix = np.full((num_nodes, num_nodes), np.inf)

        for edge, attr in self.hypergraph.edge_weights.items():
            if isinstance(edge[0], int) and isinstance(edge[1], int):
                matrix[self.hypergraph.nodes.index(edge[0]), self.hypergraph.nodes.index(edge[1])] = attr['weight']
                if attr['attributes']['Por'] == 0:
                    matrix[self.hypergraph.nodes.index(edge[1]), self.hypergraph.nodes.index(edge[0])] = attr['weight']
        
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    matrix[i, j] = min(matrix[i, j], matrix[i, k] + matrix[k, j])
        for i, hyperedge in enumerate(self.hypergraph.hyperedges):
            max = 0

            for k in range(num_nodes):
                for j in range(num_nodes):
                    if (k + 1 in hyperedge) and (j + 1 in hyperedge) and (j != k) :
                        if matrix[k, j] > max:
                            max = matrix[k, j]
            print(self.hypergraph.hyperedge_weights)
            self.hypergraph.hyperedge_weights[hyperedge] = max

        for edge, attr in self.hypergraph.edge_weights.items():
                if (not isinstance(edge[0], int)) and (not isinstance(edge[1], int)):
                    hyperedge1, hyperedge2 = edge
                    matrix[len(self.hypergraph.nodes) + self.hypergraph.hyperedges.index(hyperedge1),
                        len(self.hypergraph.nodes) + self.hypergraph.hyperedges.index(hyperedge2)] = attr['weight']
                    if attr['attributes']['Por'] == 0:
                        matrix[len(self.hypergraph.nodes) + self.hypergraph.hyperedges.index(hyperedge2),
                            len(self.hypergraph.nodes) + self.hypergraph.hyperedges.index(hyperedge1)] = attr['weight']
                        
                elif not isinstance(edge[0], int):
                    hyperedge, node = edge
                    matrix[len(self.hypergraph.nodes) + self.hypergraph.hyperedges.index(hyperedge),
                        self.hypergraph.nodes.index(node)] = attr['weight']
                    if attr['attributes']['Por'] == 0:
                        matrix[self.hypergraph.nodes.index(node),
                            len(self.hypergraph.nodes) + self.hypergraph.hyperedges.index(hyperedge)] = attr['weight']

                elif not isinstance(edge[1], int):
                    node, hyperedge = edge
                    matrix[len(self.hypergraph.nodes) + self.hypergraph.hyperedges.index(hyperedge),
                        self.hypergraph.nodes.index(node)] = attr['weight']
                    if attr['attributes']['Por'] == 0:
                        matrix[self.hypergraph.nodes.index(node),
                            len(self.hypergraph.nodes) + self.hypergraph.hyperedges.index(hyperedge)] = attr['weight']

        for i, hyperedge in enumerate(self.hypergraph.hyperedges):
            for j in self.hypergraph.nodes:
                if j in hyperedge:  # Check If the vertex 'j' belongs to the hyperedge. (Adjust this as per your data structure)
                    matrix[len(self.hypergraph.nodes) + i, j - 1] = min(
                        matrix[len(self.hypergraph.nodes) + i, j - 1],
                        self.hypergraph.hyperedge_weights.get(hyperedge, np.inf)
                    )
                    matrix[j - 1, len(self.hypergraph.nodes) + i] = min(
                        matrix[j - 1, len(self.hypergraph.nodes) + i],
                        self.hypergraph.hyperedge_weights.get(hyperedge, np.inf)
                    )

        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    matrix[i, j] = min(matrix[i, j], matrix[i, k] + matrix[k, j])

        for i in range(num_nodes):
            matrix[i, i] = 0

        return matrix

    def create_graph(self):
        """
        Creates a graph based on the given matrix.

        Parameters:
            self (object): The current instance of the class.
        
        Returns:
            graph (nx.Graph): The created graph.
        """
        graph = nx.Graph()
        num_nodes = len(self.matrix)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.isfinite(self.matrix[i, j]):
                    graph.add_edge(i, j, weight=self.matrix[i, j])
        return graph

    def compute_centralities(self):
        """
        Calculate the centralities of the graph.

        Returns:
            eccentricities (list): A list of eccentricities for each node in the graph.
            radius (float): The radius of the graph.
            diameter (float): The diameter of the graph.
            closeness_centralities (list): A list of closeness centralities for each node in the graph.
            degree_centralities (list): A list of degree centralities for each node in the graph.
        """
        eccentricities = [np.max(distances) for distances in self.matrix]
        radius = np.min([np.max(distances) for distances in self.matrix])
        diameter = np.max(self.matrix)

        closeness_centralities = [1 / np.sum(distances) if np.sum(distances) != 0 else 0 for distances in self.matrix]
        degree_centralities = [np.sum(distances != np.inf) / (self.num_nodes - 1) for distances in self.matrix]

        return eccentricities, radius, diameter, closeness_centralities, degree_centralities

    def compute_central_and_peripheral_nodes(self):
        """
        Compute the central and peripheral nodes of the graph.
        
        Returns:
            central_nodes (list): A list of nodes that have the minimum eccentricity.
            peripheral_nodes (list): A list of nodes that have the maximum eccentricity.
        """
        eccentricities = self.compute_centralities()[0]
        min_eccentricity = np.min(eccentricities)
        max_eccentricity = np.max(eccentricities)

        central_nodes = [node for node, eccentricity in enumerate(eccentricities) if eccentricity == min_eccentricity]
        peripheral_nodes = [node for node, eccentricity in enumerate(eccentricities) if eccentricity == max_eccentricity]

        return central_nodes, peripheral_nodes

    def print_matrix(self):
        """
        Print the matrix representation of the hypergraph.

        This function prints the matrix representation of the hypergraph. It first prints the matrix as a list, 
        followed by printing the matrix as a DataFrame with labeled rows and columns. The matrix includes both 
        nodes and hyperedges as labels. 

        Parameters:
        None

        Returns:
        None
        """
        print(self.matrix)
        nodes = ["gv" + str(i) for i in self.hypergraph.nodes]
        hyperedges = ["he" + str(i) for i in range(1, len(self.hypergraph.hyperedges) + 1)]
        df = pd.DataFrame(self.matrix, index=nodes + hyperedges, columns=nodes + hyperedges)
        print(df)

    def print_eccentricities(self):
        """
        Print the eccentricities of all nodes in the hypergraph.

        This function computes the eccentricities of all nodes in the hypergraph and prints them. The eccentricity of a node in a hypergraph is the maximum distance between that node and any other node in the hypergraph. The function uses the `compute_centralities` method to calculate the eccentricities and then prints them in a formatted manner.

        Parameters:
        - None

        Returns:
        - None
        """
        eccentricities = self.compute_centralities()[0]
        print("\nEccentricities:")
        for i, eccentricity in enumerate(eccentricities):
            label = f"he{i - len(self.hypergraph.nodes) + 1}" if i >= len(self.hypergraph.nodes) else f"gv{i + 1}"
            print(f"  {label}: {eccentricity:.2f}")

    def print_centralities(self):
        """
        Print the closeness centralities and degree centralities of the hypergraph.

        This function computes the closeness centralities and degree centralities of the hypergraph using the `compute_centralities` method. It then prints the results in a formatted manner.

        Parameters:
        - None

        Returns:
        - None
        """
        closeness_centralities = self.compute_centralities()[3]
        degree_centralities = self.compute_centralities()[4]
        print("\nCloseness Centralities:")
        for i, closeness_centrality in enumerate(closeness_centralities):
            label = f"he{i - len(self.hypergraph.nodes) + 1}" if i >= len(self.hypergraph.nodes) else f"gv{i + 1}"
            print(f"  {label}: {closeness_centrality:.2f}")
        print("\nDegree Centralities:")
        for i, degree_centrality in enumerate(degree_centralities):
            label = f"he{i - len(self.hypergraph.nodes) + 1}" if i >= len(self.hypergraph.nodes) else f"gv{i + 1}"
            print(f"  {label}: {degree_centrality:.2f}")

    def print_central_and_peripheral_nodes(self):
        """
        Print the central and peripheral nodes of the hypergraph.

        This function computes the central and peripheral nodes of the hypergraph using the `compute_central_and_peripheral_nodes` method. It then converts the node indices to labels based on whether the index is greater than or equal to the number of nodes in the hypergraph. The central nodes are labeled as `"hei"` if `i` is greater than or equal to the number of nodes, otherwise they are labeled as `"gvi"`, where `i` is the index. Similarly, the peripheral nodes are labeled as `"hej"` if `j` is greater than or equal to the number of nodes, otherwise they are labeled as `"gvj"`, where `j` is the index.

        Parameters:
            self (class): The instance of the class.

        Returns:
            None
        """
        central_nodes = self.compute_central_and_peripheral_nodes()[0]
        peripheral_nodes = self.compute_central_and_peripheral_nodes()[1]
        central_labels = [f"he{i - len(self.hypergraph.nodes) + 1}" if i >= len(self.hypergraph.nodes) else f"gv{i + 1}" for i in central_nodes]
        peripheral_labels = [f"he{i - len(self.hypergraph.nodes) + 1}" if i >= len(self.hypergraph.nodes) else f"gv{i + 1}" for i in peripheral_nodes]
        print("\nCentral Nodes:", central_labels)
        print("Peripheral Nodes:", peripheral_labels)
    
    def compute_diameter_and_radius(self):
        """
        Compute the diameter and radius of the graph represented by the matrix.

        Returns:
            The diameter and radius of the graph.

        Notes:
            - The diameter of a graph is the maximum distance between any two nodes.
            - The radius of a graph is the minimum maximum distance from any node to all other nodes.
        """
        all_distances = self.matrix[np.isfinite(self.matrix)]
        diameter = np.max(all_distances)
        radius = np.min(np.max(self.matrix, axis=0))
        return diameter, radius

    def print_diameter_and_radius(self):
        """
        Print the computed diameter and radius of the object.
        
        Returns:
            tuple: A tuple containing the computed diameter and radius.
        """
        diameter, radius = self.compute_diameter_and_radius()
        print(f"\nDiameter: {diameter:.2f}")
        print(f"Radius: {radius:.2f}")

        return diameter, radius

if __name__ == "__main__":
    json_hypergraph = {
    "nodes": {
        1: {"type": 1, "weight": None},
        2: {"type": 1, "weight": None},
        3: {"type": 1, "weight": None},
        4: {"type": 1, "weight": None},
        5: {"type": 1, "weight": None},
        6: {"type": 1, "weight": None},
        7: {"type": 1, "weight": None},
        8: {"type": 1, "weight": None},
        9: {"type": 1, "weight": None},
        10: {"type": 1, "weight": None},
    },
    "edges": [
        {"nodes": (1, 2), "weight": 0.6, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (2, 3), "weight": 0.5, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (1, 4), "weight": 0.5, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (3, 4), "weight": 0.6, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (7, 8), "weight": 0.9, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (8, 9), "weight": 0.5, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (9, 10), "weight": 0.9, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (10, 7), "weight": 0.9, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": ((1, 2, 3, 4), 5), "weight": 0.2, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": ((1, 2, 3, 4), 6), "weight": 0.8, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": (5, 6), "weight": 0.7, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": ((7, 8, 9, 10), 6), "weight": 0.9, "attributes": {"Por": 0, "Pt": 0}},
        {"nodes": ((7, 8, 9, 10), (1, 2, 3, 4)), "weight": 0.6, "attributes": {"Por": 0, "Pt": 0}},
    ],
    "hyperedges": {
        (1, 2, 3, 4): {"type": 1},
        (7, 8, 9, 10): {"type": 1},
    },
    }

# Преобразование JSON в объект Hypergraph
    hypergraph_from_json = HBellmanFord(**json_hypergraph)

    metrics_calculator = HypergraphMetrics(hypergraph_from_json)

    metrics_calculator.print_matrix()

    metrics_calculator.print_eccentricities()

    metrics_calculator.print_centralities()

    metrics_calculator.print_central_and_peripheral_nodes()

    metrics_calculator.print_diameter_and_radius()

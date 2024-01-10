"""Module containing HBellmanFord, HypergraphMetrics testing"""

import pytest
import numpy as np
from raw_bellman_ford.algorithms.h_bellman_ford import HBellmanFord, HypergraphMetrics

@pytest.fixture
def json_hypergraph():
    """
    Fixture that returns a JSON hypergraph.
    
    Returns:
        dict: A dictionary representing the JSON hypergraph.
    """
    return {
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

def test_hypergraph_creation(json_hypergraph):
    """
    Generate a function comment for the given function body.

    Parameters:
        json_hypergraph (dict): A dictionary representing a hypergraph.

    Returns:
        None
    """
    hypergraph = HBellmanFord(**json_hypergraph)
    assert isinstance(hypergraph, HBellmanFord)

def test_metrics_calculation(json_hypergraph):
    """
    Calculates the metrics for a given hypergraph.

    Args:
        json_hypergraph (dict): A dictionary representing the hypergraph.

    Returns:
        None
    """
    hypergraph = HBellmanFord(**json_hypergraph)
    metrics_calculator = HypergraphMetrics(hypergraph)
    assert isinstance(metrics_calculator, HypergraphMetrics)

    assert metrics_calculator.matrix is not None


def test_printing_methods(json_hypergraph, capsys):
    """
    Generates the function comment for the given function body.

    Args:
        json_hypergraph (dict): A dictionary representing the hypergraph.
        capsys (object): The capsys object used for capturing the output.

    Returns:
        None
    """
    hypergraph = HBellmanFord(**json_hypergraph)
    metrics_calculator = HypergraphMetrics(hypergraph)

    metrics_calculator.print_matrix()
    captured = capsys.readouterr()

def test_compute_diameter_and_radius(json_hypergraph):
    """
    Compute the diameter and radius of a hypergraph.

    Parameters:
        json_hypergraph (dict): A dictionary representing a hypergraph.

    Raises:
        AssertionError: If the computed diameter or radius does not match the expected values.
    """
    hypergraph = HBellmanFord(**json_hypergraph)
    metrics_calculator = HypergraphMetrics(hypergraph)

    diameter, radius = metrics_calculator.compute_diameter_and_radius()
    assert np.isclose(diameter, 3.10, rtol=1e-2)
    assert np.isclose(radius, 1.7, rtol=1e-2)

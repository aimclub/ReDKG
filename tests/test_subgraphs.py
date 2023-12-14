import torch
import pytest
from redkg.generate_subgraphs import generate_subgraphs, generate_subgraphs_dataset


def test_generate_subgraphs():
    """
    Test that the generated subgraphs have the correct number of nodes and links.
    """
    dataset = {
        'nodes': [{'id': i} for i in range(10)],
        'links': [{'source': i, 'target': i + 1} for i in range(9)]
    }
    num_subgraphs = 5
    min_nodes = 2
    max_nodes = 5

    result = generate_subgraphs(dataset, num_subgraphs, min_nodes, max_nodes)

    assert len(result) == num_subgraphs
    for subgraph in result:
        assert min_nodes <= len(subgraph['nodes']) <= max_nodes
        for link in subgraph['links']:
            assert link['source'] in [node['id'] for node in subgraph['nodes']]
            assert link['target'] in [node['id'] for node in subgraph['nodes']]


@pytest.fixture
def large_dataset():
    """
    Return a mock dataset with 20 nodes and 5 features.
    """
    class MockDataset:
        def __init__(self):
            self.node_mapping = {i: i for i in range(20)}
            self.x = torch.randn(20, 5)
            self.y = torch.randn(20, 1)

    return MockDataset()


def test_generate_subgraphs_dataset(large_dataset):
    """
    Test that the generated subgraphs have the correct number of nodes and links.
    """
    subgraphs = [
        {
            'nodes': [{'id': i} for i in range(5)],
            'links': [{'source': i, 'target': i + 1} for i in range(4)]
        },
        {
            'nodes': [{'id': i + 5} for i in range(5)],
            'links': [{'source': i + 5, 'target': i + 6} for i in range(4)]
        }
    ]

    result = generate_subgraphs_dataset(subgraphs, large_dataset)

    assert len(result) == len(subgraphs)
    for data in result:
        assert data.x.shape == large_dataset.x.shape
        assert data.y.shape == large_dataset.y.shape
        assert data.edge_index.shape[0] == 2

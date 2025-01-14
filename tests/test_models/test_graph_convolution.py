import numpy as np
import pytest
import torch

from redkg.models.graph_convolution import GraphConvolution


@pytest.fixture
def mock_adj_matrix():
    """Mock adjacency matrix"""
    return torch.FloatTensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


@pytest.fixture
def mock_input():
    """Mock input tensor"""
    return torch.FloatTensor([[1, 2], [3, 4], [5, 6]])


def test_initialization(mock_adj_matrix, monkeypatch):
    """Test initialization of GraphConvolution"""

    def mock_load(*args, **kwargs):
        return mock_adj_matrix.numpy()

    monkeypatch.setattr(np, 'load', mock_load)
    layer = GraphConvolution(2, 3)

    assert layer.in_features == 2
    assert layer.out_features == 3
    assert layer.adj.shape == (3, 3)


def test_forward(mock_adj_matrix, mock_input, monkeypatch):
    """Test forward pass"""

    def mock_load(*args, **kwargs):
        return mock_adj_matrix.numpy()

    monkeypatch.setattr(np, 'load', mock_load)
    layer = GraphConvolution(2, 3)

    output = layer(mock_input)
    assert output.shape == (3, 3)
    assert isinstance(output, torch.FloatTensor)

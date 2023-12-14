from redkg.negative_samples import common_neighbors, generate_negative_samples
import torch


def test_common_neighbors():
    """
    Test that the common neighbors are correctly computed.
    """

    # node connectivity: 0-1, 1-2, 2-3, 3-0
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    neighbors = common_neighbors(edge_index, num_nodes=4)

    assert neighbors == {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {0, 2}}


def test_generate_negative_samples():
    """
    Test that the negative samples are correctly generated.
    """

    # node connectivity: 0-1, 1-2, 2-3, 3-0
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    negative_samples = generate_negative_samples(edge_index, num_nodes=4, num_neg_samples=1)

    for ns in negative_samples:
        assert len(ns) == 2
        assert ns[0] != ns[1]
        assert ns not in edge_index.tolist()

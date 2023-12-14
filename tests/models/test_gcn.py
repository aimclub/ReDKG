import torch
import torch.nn.functional as F
from redkg.models.gcn import GCN


def test_GCN():
    """
    Test that the GCN model can be instantiated and run.
    """
    model = GCN(
        in_channels=100,
        hidden_channels=200,
        out_channels=50,
        num_layers=3,
        activation=F.relu,
        dropout_rate=0.5
    )

    x = torch.randn(16, 100)  # random node feature matrix of shape [num_nodes, in_channels]

    # edge_index: COO format graph adjacency matrix, shape [2, num_edges]
    edge_index = torch.randint(high=16, size=(2, 48), dtype=torch.long)

    output = model(x, edge_index)

    # Check the output shape
    assert output.shape == (16, 50)

    # Check the forward pass
    assert not torch.isnan(output).any()

import torch

from redkg.models.basic_models import MLP, Net


def test_Net():
    """
    Test that the Net model can be instantiated and run.
    """
    model = Net()

    x = torch.randn(16, 20)  # random node feature matrix of shape [num_nodes, in_channels]

    output = model(x)

    # Check the output shape
    assert output.shape == (16, 3)

    # Check the forward pass
    assert not torch.isnan(output).any()

    assert (output > 1).sum() == 0
    assert (output < -1).sum() == 0


def test_MLP():
    """
    Test that the MLP model can be instantiated and run.
    """
    model = MLP(
        input_shape=20,
        encoded_size=16,
        arch=[32, 64, 32]
    )

    x = torch.randn(16, 20)  # random node feature matrix of shape [num_nodes, in_channels]

    output = model(x)

    # Check the output shape
    assert output.shape == (16, 32)

    # Check the forward pass
    assert not torch.isnan(output).any()

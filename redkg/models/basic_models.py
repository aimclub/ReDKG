from typing import List

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Net(nn.Module):
    """Network Layer"""

    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Linear(20, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dense3 = nn.Linear(128, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Forward

        :param x: Tensor
        :returns: Tensor
        """
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self, input_shape: int, encoded_size: int, arch: List[int]) -> None:
        super().__init__()

        layers = []
        inputs = input_shape
        for arc in arch:
            layers.append(nn.Linear(inputs, arc))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(arc))
            inputs = arc
        layers.append(nn.Linear(arch[-1], encoded_size * 2))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward

        :param x: Tensor
        :returns: Tensor
        """
        return self.encoder(x)

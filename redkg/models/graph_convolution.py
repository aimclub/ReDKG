import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class GraphConvolution(nn.Module):
    """GCN graph convolution layer"""

    def __init__(self, in_features: int, out_features: int) -> None:
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features)
        """
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        """
        self.adj = torch.FloatTensor(np.load("./data/movie/kg_adj_mat.npy"))

    def reset_parameters(self) -> None:
        """Reset model parameters"""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor) -> Tensor:
        """Forward

        :param x: Tensor
        :returns: Tensor
        """
        # support = torch.mm(input, self.weight)
        support = self.weight(x)
        output = torch.spmm(self.adj, support)
        return output
        """
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        """

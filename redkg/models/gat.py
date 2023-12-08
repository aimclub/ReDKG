import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """Graph Attention Network with customizable layers and activation"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, activation=torch.relu, dropout_rate=0.5, heads=1):
        super(GAT, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.convs = torch.nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate))
        for _ in range(num_layers - 2):  # Subtract 2 to account for the first and last layers
            # In GAT, input and output dimensions must be multiplied by the number of heads
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_rate))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate))

        # Edge predictor
        self.edge_predictor = nn.Sequential(
            # Input size - doubled, because we combine pairs of nodes
            nn.Linear(2 * out_channels, hidden_channels),

            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # Output size - 1, for predicting the probability of a link
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index):
        """Forward pass through the model"""
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def predict_edges(self, edge_embeddings):
        """Predict the probability of a link between pairs of nodes"""
        return self.edge_predictor(edge_embeddings)

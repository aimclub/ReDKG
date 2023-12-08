import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """Graph SAmple and aggreGatE"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, activation=torch.relu, dropout_rate=0.5):
        super(GraphSAGE, self).__init__()
        self.activation = activation
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.convs = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):  # Subtract 2 to account for the first and last layers
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return x

    def predict_edges(self, edge_embeddings):
        """Predict the probability of a link between pairs of nodes"""
        return self.edge_predictor(edge_embeddings)

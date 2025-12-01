import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, AttentionalAggregation, LayerNorm
import torch

class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        # self.bn1 = nn.BatchNorm1d(hidden_channels)
        # self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.att_gate = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

        self.att_pool = AttentionalAggregation(gate_nn=self.att_gate)

        self.classifier = nn.Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        # x = self.bn1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        # x = self.bn2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # x = global_add_pool(x, batch)
        x = self.att_pool(x, batch)
        x = self.classifier(x)
        return x

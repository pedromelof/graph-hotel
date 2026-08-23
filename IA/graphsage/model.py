from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        if num_layers not in (2, 3):
            raise ValueError("num_layers deve ser 2 ou 3")
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class LinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z_src, z_dst], dim=-1)
        h = F.relu(self.lin1(h))
        return self.lin2(h).squeeze(-1)


class GraphSAGELinkPredictionModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.decoder = LinkPredictor(out_channels)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        return self.decoder(z[src], z[dst])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

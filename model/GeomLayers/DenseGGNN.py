# model/GeomLayers/DenseGGNN.py

import torch
import torch.nn as nn
from typing import Optional
from torch_geometric.nn.conv.gated_graph_conv import GatedGraphConv
from torch_geometric.utils import dense_to_sparse

class DenseGGNN(nn.Module):
    """
    Dense implementation of Gated Graph Neural Networks.
    Handles batched dense adjacency matrices by converting to sparse format internally.
    """
    def __init__(self, out_channels: int, num_layers: int = 1):
        super(DenseGGNN, self).__init__()
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.gnn = GatedGraphConv(
            out_channels=out_channels,
            num_layers=num_layers,
            aggr='add',
            bias=True
        )

    def forward(
            self,
            x: torch.Tensor,    # [B, N, D]
            adj: torch.Tensor,  # [B, N, N]
            mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of DenseGGNN.

        Args:
            x: Node features tensor [B, N, D]
            adj: Adjacency matrix tensor [B, N, N]
            mask: Optional mask tensor [B, N]

        Returns:
            Updated node features [B, N, out_channels]
        """
        batch_size, num_nodes, in_channels = x.size()

        # Convert dense adjacency matrices to edge_index format
        edge_indices = []
        for i in range(batch_size):
            e_idx, _ = dense_to_sparse(adj[i])
            if e_idx.numel() > 0:
                e_idx = e_idx + i * num_nodes
            edge_indices.append(e_idx)

        # Concatenate edge indices
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)

        # Reshape node features
        x_flat = x.reshape(batch_size * num_nodes, in_channels)

        # Apply GNN
        output = self.gnn(x_flat, edge_index)

        # Reshape output
        output = output.reshape(batch_size, num_nodes, self.out_channels)

        # Apply mask if provided
        if mask is not None:
            output = output * mask.unsqueeze(-1)

        return output

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        self.gnn.reset_parameters()

    @staticmethod
    def get_sparse_adj(
            adj: torch.Tensor,
            batch_size: int,
            num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        edge_indices = []
        edge_weights = []

        for i in range(batch_size):
            edge_index, weight = dense_to_sparse(adj[i])
            if edge_index.size(1) > 0:
                edge_index = edge_index + i * num_nodes
            edge_indices.append(edge_index)
            edge_weights.append(weight)

        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_weight = torch.cat(edge_weights, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=adj.device)
            edge_weight = torch.empty((0,), dtype=adj.dtype, device=adj.device)

        return edge_index, edge_weight
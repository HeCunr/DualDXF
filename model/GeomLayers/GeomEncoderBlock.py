# model/GeomLayers/GeomEncoderBlock.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GeomLayers.NodeAggregator import NodeAggregator

class GeomEncoderBlock(nn.Module):
    def __init__(self, d_model: int, in_nodes: int, out_nodes: int, num_ggnn_layers: int = 1):
        super().__init__()
        self.aggregator = NodeAggregator(d_model, in_nodes, out_nodes)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj, mask):
        """
        x:   [B, N, d_model]
        adj: [B, N, N]
        mask:[B, N]

        return:
          pfeat: [B, out_nodes, d_model]
          padj : [B, out_nodes, out_nodes]
          pmask: [B, out_nodes]
        """
        pfeat, padj, pmask = self.aggregator(x, adj, mask)
        pfeat = self.norm(pfeat)
        pfeat = F.relu(pfeat)
        return pfeat, padj, pmask

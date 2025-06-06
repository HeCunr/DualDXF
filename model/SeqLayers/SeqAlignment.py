# model/SeqLayers/SeqAlignment.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SeqLayers.seq_transformer_encoder import TwoLayerMLP

class NodeAlignmentHead(nn.Module):

    def __init__(self, d_model: int, alignment='concat',  latent_dropout=0.1):
        super().__init__()
        self.alignment = alignment
        if alignment.lower() == 'concat':
            self.alignment_layer = nn.Linear(2 * d_model, d_model)
        else:
            raise NotImplementedError(f"Unknown alignment={alignment}")

        self.projection = TwoLayerMLP(hidden_dim=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(latent_dropout)

    def perform_alignment(self, z_view1: torch.Tensor, z_view2: torch.Tensor):

        attention = self.node_alignment_attention(z_view1, z_view2)  # [B,N,N]
        att_v2 = torch.bmm(attention, z_view2)  # [B,N,d_model]
        att_v1 = torch.bmm(attention.transpose(1, 2), z_view1)  # [B,N,d_model]

        if self.alignment.lower() == 'concat':
            merged1 = torch.cat([z_view1, att_v2], dim=-1)  # [B,N,2d]
            merged2 = torch.cat([z_view2, att_v1], dim=-1)  # [B,N,2d]
            out1 = self.alignment_layer(merged1)  # => [B,N,d_model]
            out2 = self.alignment_layer(merged2)
        else:
            raise NotImplementedError()
        B, N, d = out1.shape
        out1_flat = out1.reshape(B * N, d)
        out2_flat = out2.reshape(B * N, d)

        out1_flat = self.projection(out1_flat)  # => (B*N, d)
        out1_flat = self.bn(out1_flat)
        out1_flat = self.dropout(out1_flat)

        out2_flat = self.projection(out2_flat)
        out2_flat = self.bn(out2_flat)
        out2_flat = self.dropout(out2_flat)

        out1 = out1_flat.view(B, N, d)
        out2 = out2_flat.view(B, N, d)

        return out1, out2

    def node_alignment_attention(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        v1_norm = F.normalize(v1, dim=-1)   # [B,N,d]
        v2_norm = F.normalize(v2, dim=-1)   # [B,N,d]
        att = torch.bmm(v1_norm, v2_norm.transpose(1,2))  # [B,N,N]
        att = F.softmax(att, dim=-1)
        return att

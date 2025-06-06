# model/GeomLayers/GeomAlignment.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAlignmentHead(nn.Module):

    def __init__(self, d_model: int, alignment, perspectives, tau=0.07):
        super().__init__()
        self.d_model = d_model
        self.alignment = alignment
        self.perspectives = perspectives
        self.tau = tau
        if alignment.lower() == 'concat':
            self.alignment_layer = nn.Linear(2*d_model, d_model)
        else:
            raise NotImplementedError(f"Unknown alignment={alignment}")

        hidden_dim = d_model * 2
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

        self.ce = nn.CrossEntropyLoss()

    def perform_alignment(self, z_view1: torch.Tensor, z_view2: torch.Tensor):

        attention = self.node_alignment_attention(z_view1, z_view2)  # [B,N,N]

        att_v2 = torch.bmm(attention, z_view2)  # [B,N,d_model]
        att_v1 = torch.bmm(attention.transpose(1,2), z_view1)

        if self.alignment.lower() == 'concat':
            merged1 = torch.cat([z_view1, att_v2], dim=-1)  # [B,N,2d]
            merged2 = torch.cat([z_view2, att_v1], dim=-1)
            out1 = self.alignment_layer(merged1)  # => [B,N,d_model]
            out2 = self.alignment_layer(merged2)
            return out1, out2
        else:
            raise NotImplementedError()

    def node_alignment_attention(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        v1_norm = F.normalize(v1, dim=-1)   # [B,N,d]
        v2_norm = F.normalize(v2, dim=-1)   # [B,N,d]
        att = torch.bmm(v1_norm, v2_norm.transpose(1,2))  # [B,N,N]
        att = F.softmax(att, dim=-1)
        return att


    def loss(self, z1: torch.Tensor, z2: torch.Tensor):

        B, N, d = z1.shape
        device = z1.device

        p1 = self.proj_head(z1)  # [B,N,d]
        p2 = self.proj_head(z2)  # [B,N,d]

        p1 = p1.view(B*N, d)
        p2 = p2.view(B*N, d)

        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)

        features = torch.cat([p1, p2], dim=0)  # => (2*B*N, d)
        total_size = features.size(0)         # 2*B*N

        similarity_matrix = torch.matmul(features, features.t())

        label_matrix = torch.zeros(total_size, total_size, device=device, dtype=torch.bool)

        half = B*N
        eye_mat = torch.eye(half, dtype=torch.bool, device=device)
        label_matrix[:half, half:] = eye_mat
        label_matrix[half:, :half] = eye_mat

        mask = torch.eye(total_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix[~mask].view(total_size, -1)
        label_matrix = label_matrix[~mask].view(total_size, -1)

        positives = similarity_matrix[label_matrix]
        negatives = similarity_matrix[~label_matrix]

        positives = positives.view(total_size, 1)
        negatives = negatives.view(total_size, -1)
        logits = torch.cat([positives, negatives], dim=1)  # => [total_size, 1 + X]

        labels = torch.zeros(total_size, dtype=torch.long, device=device)

        logits = logits / self.tau
        cl_loss = self.ce(logits, labels)

        return cl_loss

#model/GeomLayers/Geom_embedding.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeomEmbedding(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.num_entity_types = 12
        self.embedding_type = nn.Embedding(
            num_embeddings=self.num_entity_types,
            embedding_dim=d_model
        )
        self.param_input_dim = 45
        self.param_linear = nn.Linear(self.param_input_dim, d_model)
        self.pos_linear = nn.Linear(2, d_model)

    def forward(self, features: torch.Tensor, pos2d: torch.Tensor):

        entity_type = features[:, :, 0]
        entity_type = entity_type.long().clamp_(0, self.num_entity_types - 1)

        # => [B, N, d_model]
        e_type = self.embedding_type(entity_type)

        #   param_matrix: [B, N, 45]
        param_matrix = features[:, :, 1:]  # float
        B, N, P = param_matrix.shape  # P=45
        #  reshape => [B*N, 45]
        param_flat = param_matrix.view(B*N, P)
        # => linear => [B*N, d_model]
        e_param_flat = self.param_linear(param_flat)
        # => reshape => [B, N, d_model]
        e_param = e_param_flat.view(B, N, self.d_model)

        #   [B, N, 2] => [B*N,2] => linear => [B*N,d_model]
        pos_flat = pos2d.view(B*N, 2)
        e_pos_flat = self.pos_linear(pos_flat)
        e_pos = e_pos_flat.view(B, N, self.d_model)

        e = e_type + e_param + e_pos  # [B, N, d_model]
        return e

# model/SeqLayers/Seq_embedding.py

import torch
import torch.nn as nn

class PositionalEncodingLUT(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):

        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x = x + self.pos_embed(positions)  # (batch_size, seq_len, d_model)
        return self.dropout(x)

class SeqEmbedding(nn.Module):

    def __init__(self, d_model, max_len, num_entity_types=13, num_params=45):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.num_entity_types = num_entity_types
        self.num_params = num_params

        self.entity_embed = nn.Embedding(num_entity_types, d_model)

        self.param_fc = nn.Linear(num_params, d_model)

        self.pos_encoding = PositionalEncodingLUT(d_model, dropout=0.1, max_len=max_len)

    def forward(self, entity_type, entity_params):

        safe_entity_type = entity_type.clone()
        safe_entity_type[safe_entity_type < 0] = self.num_entity_types - 1  # 即 12
        safe_entity_type.clamp_(0, self.num_entity_types - 1)

        entity_type_embed = self.entity_embed(safe_entity_type)  # (B, seq_len, d_model)

        safe_params = entity_params.clone()
        safe_params[safe_params < 0] = 0.0
        entity_params_embed = self.param_fc(safe_params)  # (B, seq_len, d_model)

        x = entity_type_embed + entity_params_embed
        x = self.pos_encoding(x)
        return x

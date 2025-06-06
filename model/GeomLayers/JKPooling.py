# model/GeomLayers/JKPooling.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class JKSumConcatProject(nn.Module):

    def __init__(self, d_model, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.linear_merge = nn.Linear(num_layers * d_model, d_model)
        self.act_merge = nn.ReLU()

    def forward(self, all_layer_nodes, mask=None):

        pooled_list = []
        for x_l in all_layer_nodes:
            if mask is not None:
                x_l = x_l * mask.unsqueeze(-1)
            layer_sum = x_l.sum(dim=1)         # [B, D]
            pooled_list.append(layer_sum)

        cat_feat = torch.cat(pooled_list, dim=-1)
        out = self.linear_merge(cat_feat)
        out = self.act_merge(out)
        norm = torch.norm(out, p=2, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        out = out / norm
        return out

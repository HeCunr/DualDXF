# model/GeomLayers/Geom_association_diffusion.py

import torch
import torch.nn.functional as F

def multi_scale_graph_diffusion(adj: torch.Tensor, scales=[1,2,3], alpha_scales=None):

    device = adj.device
    B, N, _ = adj.size()

    row_sum = adj.sum(dim=-1, keepdim=True)  # [B, N, 1]
    denom = torch.clamp(row_sum, min=1e-9)
    P = adj / denom  # [B, N, N]

    if alpha_scales is None:
        alpha_scales = [1.0]*len(scales)

    diff_list = []
    for i, k in enumerate(scales):
        P_k = P.clone()
        for _ in range(k-1):
            P_k = torch.bmm(P_k, P)  # [B,N,N] * [B,N,N]
        diff_list.append(alpha_scales[i] * P_k)

    weight_sum = sum(alpha_scales)
    graph_diff = sum(diff_list) / weight_sum
    return graph_diff


def multi_scale_feature_diffusion(z: torch.Tensor, scales=[0.05, 0.1, 0.2]):

    B, N, D = z.size()
    z_norm = F.normalize(z, dim=-1)
    sim_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2))  # [B,N,N]

    diff_mats = []
    for temp in scales:
        diff_mat = F.softmax(sim_matrix / temp, dim=-1)  # [B,N,N]
        diff_mats.append(diff_mat)
    feat_diff = sum(diff_mats) / len(diff_mats)
    return feat_diff

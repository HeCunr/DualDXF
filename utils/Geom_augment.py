# utils/Geom_augment.py
import numpy as np
import torch

def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:

    B, N, F = x.size()
    drop_mask = torch.rand((B, 1, F), device=x.device) < drop_prob
    drop_mask[:, :, 0] = False
    x = x.masked_fill(drop_mask, 0)
    return x

def aug_random_edge(adj: torch.Tensor, drop_percent: float) -> torch.Tensor:

    B, N, _ = adj.shape
    drop_p = drop_percent / 2

    edge_mask = (adj > 0).float()  # [B, N, N]
    drop_mask = torch.rand_like(adj, device=adj.device) < drop_p
    b = adj * (1 - drop_mask * edge_mask)

    drop_num = edge_mask.sum(dim=[1, 2]) - (b > 0).sum(dim=[1, 2])  # [B]
    total_potential = N * N - (b > 0).sum(dim=[1, 2])  # [B]
    mask_p = drop_num / total_potential.clamp(min=1e-8)  # [B]
    add_mask = (torch.rand_like(adj, device=adj.device) < mask_p.view(B, 1, 1)) & (b == 0)
    new_adj = b + add_mask.float()

    return new_adj

def drop_pos2d(pos2d: torch.Tensor, drop_prob: float) -> torch.Tensor:

    B, N, _ = pos2d.shape

    drop_mask = torch.rand((B, N, 1), device=pos2d.device) < drop_prob

    pos2d = pos2d.masked_fill(drop_mask, 0.0)
    return pos2d



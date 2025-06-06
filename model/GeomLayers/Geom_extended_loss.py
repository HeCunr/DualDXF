# model/GeomLayers/Geom_extended_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.GeomLayers.Geom_association_diffusion import (
    multi_scale_graph_diffusion,
    multi_scale_feature_diffusion
)

class GeomExtendedContrastiveLoss(nn.Module):

    def __init__(
            self,
            alpha=0.5,
            top_k=1,
            scales_graph=[1,2],
            scales_feat=[0.05,0.1,0.2],
            tau=0.07
    ):

        super().__init__()
        self.alpha = alpha
        self.top_k = top_k
        self.scales_graph = scales_graph
        self.scales_feat = scales_feat
        self.tau = tau

    def forward(self, z_view1, z_view2, adj1, adj2, mask1=None, mask2=None):

        device = z_view1.device
        B, N, D = z_view1.shape

        G1 = multi_scale_graph_diffusion(adj1, scales=self.scales_graph)  # [B, N, N]
        G2 = multi_scale_graph_diffusion(adj2, scales=self.scales_graph)  # [B, N, N]

        S1 = multi_scale_feature_diffusion(z_view1, scales=self.scales_feat)  # [B, N, N]
        S2 = multi_scale_feature_diffusion(z_view2, scales=self.scales_feat)  # [B, N, N]

        score1 = self.alpha * G1 + (1 - self.alpha) * S1  # [B, N, N]
        score2 = self.alpha * G2 + (1 - self.alpha) * S2  # [B, N, N]

        all_losses = []

        for b_idx in range(B):
            valid_indices = None
            if (mask1 is None) and (mask2 is None):
                valid_indices = torch.arange(2*N, device=device)
            else:
                valid_flag = []
                if mask1 is not None:
                    v1_flags = mask1[b_idx].bool()
                else:
                    v1_flags = torch.ones(N, dtype=torch.bool, device=device)
                if mask2 is not None:
                    v2_flags = mask2[b_idx].bool()
                else:
                    v2_flags = torch.ones(N, dtype=torch.bool, device=device)
                valid_flag = torch.cat([v1_flags, v2_flags], dim=0)  # [2N]
                valid_indices = valid_flag.nonzero(as_tuple=True)[0]  # indices of True

            z_b1 = z_view1[b_idx]  # [N, D]
            z_b2 = z_view2[b_idx]  # [N, D]
            score1_b = score1[b_idx]  # [N, N]
            score2_b = score2[b_idx]  # [N, N]

            z_cat = torch.cat([z_b1, z_b2], dim=0)
            z_cat_norm = F.normalize(z_cat, dim=1)
            sim_mat_b = torch.mm(z_cat_norm, z_cat_norm.t())  # [2N, 2N]

            # exponentiate
            exp_sim_b = torch.exp(sim_mat_b / self.tau)

            for i in valid_indices:
                if i < N:
                    if (mask1 is not None) and (mask1[b_idx, i].item() == 0):
                        continue

                    strong_pos_idx = i + N

                    row_score = score2_b[i]  # [N], i in view2
                    topk_indices = torch.topk(row_score, self.top_k).indices  # [top_k]
                    halfpos_idx = [p.item() + N for p in topk_indices]

                else:
                    i_in_v2 = i - N
                    if (mask2 is not None) and (mask2[b_idx, i_in_v2].item() == 0):
                        continue

                    strong_pos_idx = i_in_v2
                    row_score = score1_b[i_in_v2]  # [N]
                    topk_indices = torch.topk(row_score, self.top_k).indices
                    halfpos_idx = [p.item() for p in topk_indices]

                sum_all = exp_sim_b[i].sum()
                denominator = sum_all - exp_sim_b[i, i]

                numerator = 0.0
                numerator += 1.0 * exp_sim_b[i, strong_pos_idx]

                for hp in halfpos_idx:
                    if hp == strong_pos_idx:
                        continue
                    if i < N:
                        p_in_v2 = hp - N
                        w_ip = row_score[p_in_v2]
                    else:
                        p_in_v1 = hp
                        w_ip = row_score[p_in_v1]

                    numerator += w_ip * exp_sim_b[i, hp]

                eps = 1e-9
                loss_i = -torch.log((numerator + eps) / (denominator + eps))
                all_losses.append(loss_i)

        if len(all_losses) == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(all_losses).mean()

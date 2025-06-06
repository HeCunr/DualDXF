# model/SeqLayers/Seq_extended_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SeqLayers.Seq_association_diffusion import multi_scale_diffusion

class SeqExtendedContrastiveLoss(nn.Module):


    def __init__(self, cfg, device, batch_size, temperature=0.07):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size
        self.weights = cfg.loss_weights

        self.extend_top_k = cfg.extend_top_k
        self.alpha = cfg.alpha
        self.scales = cfg.scales
        self.local_k = cfg.local_k
        self.sigma = cfg.sigma

    def forward(self, outputs):

        proj_z1 = outputs["proj_z1"]  # [B, N, D]
        proj_z2 = outputs["proj_z2"]  # [B, N, D]

        loss = self._simclr_style_loss(proj_z1, proj_z2)
        loss_contrastive = self.weights["loss_cl_weight"] * loss
        return {"loss_contrastive": loss_contrastive}

    def _build_distance_weight_matrix(self, N, sigma, device):

        indices = torch.arange(N, dtype=torch.float32, device=device)
        i_grid, p_grid = torch.meshgrid(indices, indices, indexing='ij')  # [N, N]
        distances = (i_grid - p_grid).abs()  # [N, N]
        pos_weight = torch.exp(-distances.pow(2) / (2 * sigma * sigma))  # [N, N]
        return pos_weight

    def _simclr_style_loss(self, z_view1, z_view2):

        B, N, D = z_view1.size()

        assoc_v1 = multi_scale_diffusion(z_view1, scales=self.scales)  # [B, N, N]
        assoc_v2 = multi_scale_diffusion(z_view2, scales=self.scales)  # [B, N, N]

        pos_w = self._build_distance_weight_matrix(N, self.sigma, z_view1.device)  # [N, N]
        pos_w = pos_w.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        all_losses = []

        for b_idx in range(B):
            z_b1 = z_view1[b_idx]  # [N, D]
            z_b2 = z_view2[b_idx]  # [N, D]

            z_cat = torch.cat([z_b1, z_b2], dim=0)  # [2N, D]

            z_cat_norm = F.normalize(z_cat, dim=1)
            sim_b = torch.mm(z_cat_norm, z_cat_norm.t())  # [2N, 2N]

            # exponentiate
            exp_sim_b = torch.exp(sim_b / self.temperature)  # [2N, 2N]

            assoc1_b = assoc_v1[b_idx]  # [N, N]
            assoc2_b = assoc_v2[b_idx]  # [N, N]
            pos_w_b = pos_w[b_idx]      # [N, N]

            for i in range(2*N):

                if i < N:
                    strong_positive_idx = i + N
                    score_v2_i = self.alpha * pos_w_b[i] + (1 - self.alpha) * assoc2_b[i]
                    topk_v2 = torch.topk(score_v2_i, self.extend_top_k).indices  # [K]
                    halfpos_indices = [p.item() + N for p in topk_v2]

                else:
                    i_in_v2 = i - N
                    strong_positive_idx = i_in_v2
                    score_v1_i = self.alpha * pos_w_b[i_in_v2] + (1 - self.alpha) * assoc1_b[i_in_v2]
                    topk_v1 = torch.topk(score_v1_i, self.extend_top_k).indices
                    halfpos_indices = [p.item() for p in topk_v1]

                denominator = exp_sim_b[i].sum() - exp_sim_b[i, i]
                numerator = 0.0
                numerator += 1.0 * exp_sim_b[i, strong_positive_idx]
                if i < N:
                    for hp_idx in halfpos_indices:
                        p_in_z2 = hp_idx - N
                        if hp_idx == strong_positive_idx:
                            continue
                        w_ip = score_v2_i[p_in_z2]  # half positive weight
                        numerator += w_ip * exp_sim_b[i, hp_idx]
                else:
                    for hp_idx in halfpos_indices:
                        p_in_z1 = hp_idx
                        if hp_idx == strong_positive_idx:
                            continue
                        w_ip = score_v1_i[p_in_z1]
                        numerator += w_ip * exp_sim_b[i, hp_idx]

                eps = 1e-9
                loss_i = -torch.log(numerator / (denominator + eps) + eps)
                all_losses.append(loss_i)

        losses_tensor = torch.stack(all_losses, dim=0)
        return losses_tensor.mean()

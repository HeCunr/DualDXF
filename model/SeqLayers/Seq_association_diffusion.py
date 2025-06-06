# model/SeqLayers/Seq_association_diffusion.py

import torch
import torch.nn.functional as F

def multi_scale_diffusion(z, scales=[0.05, 0.1, 0.2]):

    z_norm = F.normalize(z, dim=-1)  # [B,N,D]
    sim_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2))  # [B,N,N]

    diff_matrices = []
    for temp in scales:
        diff_matrix = F.softmax(sim_matrix / temp, dim=-1)  # [B,N,N]
        diff_matrices.append(diff_matrix)

    return sum(diff_matrices) / len(diff_matrices)

def build_association_diffusion_matrix(z, temp=0.1):

    z_norm = F.normalize(z, dim=-1)  # [B,N,D]
    sim_matrix = torch.bmm(z_norm, z_norm.transpose(1, 2))  # [B,N,N]
    return F.softmax(sim_matrix / temp, dim=-1)

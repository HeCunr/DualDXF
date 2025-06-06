# model/Seq_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqContrastiveLoss(nn.Module):
    def __init__(self, cfg, device, batch_size, temperature=0.07):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.batch_size = batch_size
        self.weights = cfg.loss_weights

    def forward(self, outputs):

        logits, labels = self._info_nce_loss(outputs["proj_z1"], outputs["proj_z2"])
        loss_contrastive = self.weights["loss_cl_weight"] * torch.nn.CrossEntropyLoss()(logits, labels)
        return {"loss_contrastive": loss_contrastive}

    def _info_nce_loss(self, f1, f2):

        def replace_nan_inf(x):
            x = x.clone()
            x[torch.isnan(x)] = 0.0
            x[torch.isinf(x)] = 0.0
            return x

        N, S, D = f1.shape
        f1 = f1.reshape(N*S, D)
        f2 = f2.reshape(N*S, D)
        f1 = replace_nan_inf(f1)
        f2 = replace_nan_inf(f2)

        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)

        features = torch.cat([f1, f2], dim=0)  # => (2*N*S, d_model)
        total_size = features.size(0)

        similarity_matrix = torch.matmul(features, features.T)

        label_matrix = torch.zeros(total_size, total_size, device=self.device)
        eye_mat = torch.eye(N*S, device=self.device)
        label_matrix[:N*S, N*S:] = eye_mat
        label_matrix[N*S:, :N*S] = eye_mat

        mask = torch.eye(total_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix[~mask].view(total_size, -1)
        label_matrix = label_matrix[~mask].view(total_size, -1)

        positives = similarity_matrix[label_matrix.bool()].view(total_size, -1)
        negatives = similarity_matrix[~label_matrix.bool()].view(total_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        logits = logits / self.temperature

        return logits, labels

#model/GeomSeqLoss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeomSeqLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, geom_repr, seq_repr):

        logits = torch.matmul(geom_repr, seq_repr.t())  # [B,B]
        logits = logits / self.temperature

        labels = torch.arange(geom_repr.size(0), dtype=torch.long, device=geom_repr.device)
        loss_i = F.cross_entropy(logits,     labels)  # geom->seq
        loss_j = F.cross_entropy(logits.t(), labels)  # seq->geom
        return 0.5*(loss_i + loss_j)

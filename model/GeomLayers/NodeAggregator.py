#model/GeomLayers/NodeAggregator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeAggregator(nn.Module):
    def __init__(self,
                 in_features: int,
                 in_nodes: int,
                 out_nodes: int,
                 threshold: float = 0.01,
                 topk_ratio: float = 0.7):

        super().__init__()
        self.in_features = in_features
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.threshold = threshold
        self.topk_ratio = topk_ratio
        self.assign_main = nn.Sequential(
            nn.Linear(in_features, out_nodes),
            nn.ReLU(),
            nn.Linear(out_nodes, out_nodes)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor):
        """
        x:   [B, N, F]
        adj: [B, N, N]
        mask:[B, N]

        Returns:
            pfeat_main: [B, out_nodes, F]
            pooled_adj: [B, out_nodes, out_nodes]
            pmask:      [B, out_nodes]
        """
        B, N, feat_dim = x.size()
        device = x.device

        x2d = x.view(B*N, feat_dim)                 # => [B*N, F]
        logits_main = self.assign_main(x2d)         # => [B*N, out_nodes]

        mask_1d = mask.view(B*N, 1)                 # [B*N,1]
        large_neg = -1e9 * (1 - mask_1d)
        logits_main = logits_main + large_neg

        assign_main_2d = F.softmax(logits_main, dim=1)     # => [B*N, out_nodes]
        S_main = assign_main_2d.view(B, N, self.out_nodes) # => [B, N, out_nodes]

        pfeat_main = torch.bmm(S_main.transpose(1, 2), x)  # => [B, out_nodes, F]

        mid = torch.bmm(adj, S_main)               # [B, N, out_nodes]
        pooled_adj = torch.bmm(S_main.transpose(1, 2), mid)  # => [B, out_nodes, out_nodes]

        actual_topk = min(self.out_nodes, int(torch.ceil(torch.tensor(self.topk_ratio * self.out_nodes)).item()))
        pooled_adj = self.threshold_topk_sparsify(pooled_adj,
                                                  threshold=self.threshold,
                                                  topk=actual_topk)

        row_sum = pooled_adj.sum(dim=-1)                # [B, out_nodes]
        d_inv_sqrt = 1.0 / (row_sum + 1e-9).sqrt()      # [B, out_nodes]

        pooled_adj = pooled_adj * d_inv_sqrt.unsqueeze(2)
        pooled_adj = pooled_adj * d_inv_sqrt.unsqueeze(1)

        pmask = torch.ones((B, self.out_nodes), device=device)

        return pfeat_main, pooled_adj, pmask

    def threshold_topk_sparsify(self,
                                adj_pool: torch.Tensor,
                                threshold: float,
                                topk: int) -> torch.Tensor:
        B, K, _ = adj_pool.shape

        mask_threshold = (adj_pool >= threshold).float()
        adj_after_thresh = adj_pool * mask_threshold

        _, indices = torch.topk(adj_after_thresh, k=topk, dim=-1)
        final_mask = torch.zeros_like(adj_pool)
        final_mask.scatter_(2, indices, 1.0)

        adj_sparse = adj_pool * final_mask
        adj_ste = adj_sparse + (adj_pool - adj_sparse).detach()

        return adj_ste
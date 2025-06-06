#model/GeomLayers/GeomEncoderStack.py
import torch
import torch.nn as nn
from model.GeomLayers.GeomEncoderBlock import GeomEncoderBlock
from model.GeomLayers.GeomGGNNBlock import GeomGGNNBlock

class GeomEncoderStack(nn.Module):


    def __init__(self, args, d_model):
        super().__init__()

        self.blocks = nn.ModuleList([
            GeomEncoderBlock(d_model, in_nodes=4096, out_nodes=1024),
            GeomEncoderBlock(d_model, in_nodes=1024, out_nodes=256),
            #GeomEncoderBlock(d_model, in_nodes=512, out_nodes=256),
            GeomEncoderBlock(d_model, in_nodes=256, out_nodes=64),
            GeomGGNNBlock(node_init_dims=d_model, args=args)  # 多层 GNN
        ])

    def forward(self, x, adj, mask):

        for block in self.blocks[:-1]:
            x, adj, mask = block(x, adj, mask)

        gg_block = self.blocks[-1]
        x_final, all_gnn_outs = gg_block(
            x, adj, mask=mask, collect_intermediate=True
        )

        return x_final, all_gnn_outs, adj, mask

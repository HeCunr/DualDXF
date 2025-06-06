#model/GeomLayers/GeomModel.py
import torch
import torch.nn as nn
from model.GeomLayers.GeomEncoderStack import GeomEncoderStack
from model.GeomLayers.Geom_embedding import GeomEmbedding
from model.GeomLayers.JKPooling import JKSumConcatProject

class GeomModel(nn.Module):
    def __init__(self, args, d_model):

        super().__init__()
        self.d_model = d_model
        self.embedding = GeomEmbedding(d_model=self.d_model)
        self.encoder_stack = GeomEncoderStack(args=args, d_model=self.d_model)
        self.jk_net = JKSumConcatProject(d_model=self.d_model, num_layers=3)

    def forward(self, features, pos2d, adj, mask):
        x = self.embedding(features, pos2d)
        x_final, all_gnn_outs, adj_enc, mask_enc = self.encoder_stack(x, adj, mask)
        x_jk = self.jk_net(all_gnn_outs, mask_enc)
        return x_final, x_jk, adj_enc, mask_enc

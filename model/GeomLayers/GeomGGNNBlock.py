#model/GeomLayers/GeomGGNNBlock.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseSAGEConv, DenseGINConv
from model.GeomLayers.DenseGGNN import DenseGGNN

class GeomGGNNBlock(nn.Module):

    def __init__(self, node_init_dims: int, args):
        super(GeomGGNNBlock, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.filters = [int(f) for f in args.filters.split('_')]
        self.num_layers = len(self.filters)
        self.last_filter = self.filters[-1]
        self.gnn_layers = self._build_gnn_layers()

    def _build_gnn_layers(self):
        layers = nn.ModuleList()
        conv_type = self.args.conv.lower()  # "gcn" / "graphsage" / "gin" / "ggnn"

        gcn_params = []
        gin_params = []
        for i in range(self.num_layers):
            in_channels = self.filters[i - 1] if i > 0 else self.args.graph_init_dim
            out_channels = self.filters[i]
            gcn_params.append(dict(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=True
            ))
            gin_mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU()
            )
            gin_params.append(dict(nn=gin_mlp))

        if conv_type == 'gcn':
            for i in range(self.num_layers):
                layer = DenseGCNConv(**gcn_params[i])
                layers.append(layer)

        elif conv_type == 'graphsage':
            for i in range(self.num_layers):
                layer = DenseSAGEConv(**gcn_params[i])
                layers.append(layer)

        elif conv_type == 'gin':
            for i in range(self.num_layers):
                layer = DenseGINConv(**gin_params[i])
                layers.append(layer)

        elif conv_type == 'ggnn':
            for i in range(self.num_layers):
                out_channels = self.filters[i]
                layer = DenseGGNN(out_channels=out_channels, num_layers=1)
                layers.append(layer)

        else:
            raise ValueError(f"Unsupported conv type: {self.args.conv}")

        return layers

    def forward(self, x, adj, mask=None, collect_intermediate=False):
        all_layers_out = []
        for i, layer in enumerate(self.gnn_layers):
            if isinstance(layer, DenseGGNN):
                out = layer(x, adj, mask=mask)
            else:
                out = layer(x, adj, mask=mask, add_loop=False)

            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            x = out

            if collect_intermediate:
                all_layers_out.append(x)

        if collect_intermediate:
            return x, all_layers_out
        else:
            return x

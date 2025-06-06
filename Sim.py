# Sim.py
import argparse
import os
import torch
import torch.nn.functional as F
import h5py
import json
import numpy as np
import torch.nn as nn
from config.config import get_config
from utils.early_stopping import EarlyStopping
from model.GeomLayers.GeomModel import GeomModel
from model.GeomLayers.GeomAlignment import NodeAlignmentHead as GeomNodeAlign
from model.SeqLayers.seq_transformer_encoder import SeqTransformer
from model.SeqLayers.SeqAlignment import NodeAlignmentHead as SeqNodeAlign
from model.SeqLayers.SelfAttPool import SelfAttPool

from dataset.dataset import GraphData, process_geom_graph
import networkx as nx

@torch.no_grad()
def load_single_geom(json_path, geom_model, device, max_n=None):

    with open(json_path, 'r') as f:
        g_info = json.load(f)

    n_num = g_info['n_num']
    g = GraphData(node_num=n_num, name=g_info['src'])
    g.features = np.array(g_info['features'], dtype=np.float32)
    if '2D-index' in g_info:
        g.pos2d = np.array(g_info['2D-index'], dtype=np.float32)
    g.adj.add_nodes_from(range(n_num))
    for u in range(n_num):
        if u < len(g_info['succs']):
            for v in g_info['succs'][u]:
                g.adj.add_edge(u, v)

    feat_mat, adj_mat, mask_mat, pos2d_mat = process_geom_graph(g, max_n=max_n)
    feat_mat  = torch.tensor(feat_mat , dtype=torch.float32).unsqueeze(0).to(device)
    adj_mat   = torch.tensor(adj_mat  , dtype=torch.float32).unsqueeze(0).to(device)
    mask_mat  = torch.tensor(mask_mat , dtype=torch.float32).unsqueeze(0).to(device)
    pos2d_mat = torch.tensor(pos2d_mat, dtype=torch.float32).unsqueeze(0).to(device)

    x_embed = geom_model.embedding(feat_mat, pos2d_mat)  # [1, N, d_model]
    x_final, all_gnn_outs, adj_enc, mask_enc = geom_model.encoder_stack(x_embed, adj_mat, mask_mat)

    geom_vec = x_final.mean(dim=1)   # => [1, d_model]
    return geom_vec


@torch.no_grad()
def load_single_seq(h5_path, seq_model, device, max_len=None):

    with h5py.File(h5_path, 'r') as hf:
        dxf_vec = hf['dxf_vec'][0]  # => shape (S, 46)
        entity_type = dxf_vec[:, 0].astype(np.int64)
        entity_param = dxf_vec[:, 1:].astype(np.float32)

    if entity_type.shape[0] > max_len:
        entity_type = entity_type[:max_len]
        entity_param = entity_param[:max_len, :]

    entity_type_t = torch.tensor(entity_type, dtype=torch.long).unsqueeze(0).to(device)
    entity_param_t = torch.tensor(entity_param, dtype=torch.float32).unsqueeze(0).to(device)

    src_embed = seq_model.embedding(entity_type_t, entity_param_t)    # => [1, seq_len, d_model]
    src_after_pool = seq_model.progressive_pool(src_embed)           # => [1, output_length, d_model]

    src_trans_in = src_after_pool.permute(1, 0, 2)                   # => [output_length, 1, d_model]
    memory = seq_model.encoder(src_trans_in)                         # => [output_length, 1, d_model]
    seq_out = memory.permute(1, 0, 2)                                # => [1, output_length, d_model]

    seq_vec = seq_out.mean(dim=1)  # => [1, d_model]
    return seq_vec


@torch.no_grad()
def encode_dxf(h5_path, json_path, geom_model, seq_model, seq_pool, Geom_Ref, Seq_Ref, device, mc_args):
    geom_vec = load_single_geom(json_path, geom_model, device, max_n=mc_args.max_n)  # [1, d_model]
    seq_vec  = load_single_seq(h5_path, seq_model, device, max_len=mc_args.max_len)      # [1, d_model]
    dxf_vec  = torch.cat([seq_vec, geom_vec], dim=-1)           # => [1, 2*d_model]

    dxf_vec = F.normalize(dxf_vec, dim=-1)  # => [1, 2*d_model]
    return dxf_vec

def main():
    parser = argparse.ArgumentParser(description="Compute similarity of two DXF files via MC model.")
    parser.add_argument('--dxf1_h5',   type=str,default="/home/vllm/DualDXF/data/Seq/SuperLFD_evaluate/....h5", help='DXF1 .h5 path')
    parser.add_argument('--dxf1_json', type=str, default="/home/vllm/DualDXF/data/Geom/SuperLFD_evaluate/....json",  help='DXF1 .json path')
    parser.add_argument('--dxf2_h5',   type=str,default="/home/vllm/DualDXF/data/Seq/SuperLFD_evaluate/....h5", help='DXF2 .h5 path')
    parser.add_argument('--dxf2_json', type=str, default="/home/vllm/DualDXF/data/Geom/SuperLFD_evaluate/....json", help='DXF2 .json path')
    parser.add_argument('--gpu_id',    type=int, default=0,     help='Which GPU to use; -1 for CPU')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu")

    mc_args = get_config()
    mc_args.save_path ="/home/vllm/DualDXF/checkpoints/Dual_best.pth"

    geom_model = GeomModel(
        args = argparse.Namespace(
            filters=mc_args.geom_filters,
            conv=mc_args.geom_conv_type,
            dropout=mc_args.geom_dropout,
            graph_init_dim=46
        ),
        d_model=mc_args.d_model
    ).to(device)

    geom_align = GeomNodeAlign(
        d_model=mc_args.d_model,
        alignment='concat',
        perspectives=mc_args.perspectives,
        tau=0.07
    ).to(device)

    seq_model = SeqTransformer(
        d_model=mc_args.seq_d_model,
        num_layers=mc_args.seq_num_layers,
        nhead=mc_args.seq_nhead,
        dim_feedforward=mc_args.seq_dim_feedforward,
        dropout=mc_args.seq_dropout,
        max_len=mc_args.max_len,
        input_length=mc_args.input_length,
        output_length=mc_args.output_length
    ).to(device)

    seq_align = SeqNodeAlign(
        d_model=mc_args.d_model,
        alignment='concat',
        latent_dropout=0.1
    ).to(device)

    seq_pool  = SelfAttPool(d_model=mc_args.d_model).to(device)

    Geom_Ref = nn.Sequential(
        nn.Linear(mc_args.d_model, mc_args.d_model),
        nn.ReLU(),
        nn.Linear(mc_args.d_model, mc_args.d_model)
    ).to(device)

    Seq_Ref = nn.Sequential(
        nn.Linear(mc_args.d_model, mc_args.d_model),
        nn.ReLU(),
        nn.Linear(mc_args.d_model, mc_args.d_model)
    ).to(device)

    stopper = EarlyStopping(
        patience=mc_args.patience,
        checkpoint_path=mc_args.save_path
    )
    stopper.load_checkpoint(
        geom_model, geom_align,
        seq_model, seq_align, seq_pool,
        Geom_Ref, Seq_Ref,
        optimizer=None,
    )

    geom_model.eval()
    seq_model.eval()
    seq_pool.eval()
    Geom_Ref.eval()
    Seq_Ref.eval()

    dxf1_vec = encode_dxf(args.dxf1_h5, args.dxf1_json, geom_model, seq_model, seq_pool, Geom_Ref, Seq_Ref, device, mc_args)  # => [1, 2*d_model]
    dxf2_vec = encode_dxf(args.dxf2_h5, args.dxf2_json, geom_model, seq_model, seq_pool, Geom_Ref, Seq_Ref, device, mc_args)

    cos_sim = F.cosine_similarity(dxf1_vec, dxf2_vec, dim=-1)
    cos_sim_01 = 0.5 * (1.0 + cos_sim.item())

    print(f"Raw Cosine Similarity = {cos_sim.item():.4f}")
    print(f"Scaled to [0,1]       = {cos_sim_01:.4f}")


if __name__ == "__main__":
    main()
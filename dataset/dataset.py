#dataset/dataset.py
import os
import json
import h5py
import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset

class GraphData:
    def __init__(self, node_num: int, name: str):
        self.node_num = node_num
        self.name = name
        self.features = None     # (n_num, 46)
        self.pos2d = None        # (n_num, 2)
        self.adj = nx.Graph()
        self.matrices = None     # (feature_mat, adj_padded, mask, pos2d_matrix)

def process_geom_graph(raw_g, max_n, feat_dim=46):

    adj_mat = nx.adjacency_matrix(raw_g.adj).toarray()
    np.fill_diagonal(adj_mat, 1.0)
    n = raw_g.node_num

    if n > max_n:
        adj_mat = adj_mat[:max_n, :max_n]
        n = max_n
        if raw_g.features.shape[0] > max_n:
            raw_g.features = raw_g.features[:max_n, :]
        if raw_g.pos2d is not None and raw_g.pos2d.shape[0] > max_n:
            raw_g.pos2d = raw_g.pos2d[:max_n, :]

    adj_padded = np.zeros((max_n, max_n), dtype=np.float32)
    adj_padded[:n, :n] = adj_mat[:n, :n]

    feature_matrix = np.zeros((max_n, feat_dim), dtype=np.float32)
    if raw_g.features is not None and raw_g.features.ndim == 1:
        raw_g.features = raw_g.features.reshape(1, -1)
    feature_matrix[:n, :] = raw_g.features[:n, :]

    mask = np.zeros((max_n,), dtype=np.float32)
    mask[:n] = 1.0

    pos2d_matrix = np.zeros((max_n, 2), dtype=np.float32)
    if raw_g.pos2d is not None:
        pos2d_matrix[:n, :] = raw_g.pos2d[:n, :]

    return feature_matrix, adj_padded, mask, pos2d_matrix

class Dataset(Dataset):

    def __init__(self, geom_data_dir, seq_data_dir, max_nodes):
        super().__init__()
        self.geom_data_dir = geom_data_dir
        self.seq_data_dir  = seq_data_dir
        self.max_nodes     = max_nodes

        self.geom_files_dict = {}
        self.seq_files_dict  = {}

        for f in os.listdir(geom_data_dir):
            if f.endswith('.json'):
                name_wo_ext = f[:-5]
                self.geom_files_dict[name_wo_ext] = os.path.join(geom_data_dir, f)

        for f in os.listdir(seq_data_dir):
            if f.endswith('.h5'):
                name_wo_ext = f[:-3]
                self.seq_files_dict[name_wo_ext] = os.path.join(seq_data_dir, f)

        common = set(self.geom_files_dict.keys()) & set(self.seq_files_dict.keys())
        self.common_filenames = sorted(list(common))

        self.samples = []
        self.fnames  = []

        for name in self.common_filenames:
            geom_path = self.geom_files_dict[name]
            seq_path  = self.seq_files_dict[name]

            valid_graph = self._load_valid_geom(geom_path)
            if valid_graph is None:
                continue

            geom_matrices = process_geom_graph(valid_graph, max_n=self.max_nodes, feat_dim=46)

            self.samples.append((geom_matrices, seq_path))
            self.fnames.append(name)

        print(f"[Dataset] total matched samples = {len(self.samples)}")

    def _load_valid_geom(self, geom_path):

        if not os.path.isfile(geom_path):
            return None

        with open(geom_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        n_num = data.get('n_num', 0)
        if n_num <= 50:
            return None
        if 'features' not in data:
            return None

        features_arr = np.array(data['features'], dtype=np.float32)
        g = GraphData(node_num=n_num, name=data.get('src', 'unknown'))
        g.features = features_arr

        # pos2d
        if '2D-index' in data:
            g.pos2d = np.array(data['2D-index'], dtype=np.float32)

        edge_count = 0
        if 'succs' in data:
            g.adj.add_nodes_from(range(n_num))
            for u in range(n_num):
                if u < len(data['succs']):
                    for v in data['succs'][u]:
                        g.adj.add_edge(u, v)
                        edge_count += 1

        if edge_count < 200:
            return None

        return g

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        geom_matrices, seq_path = self.samples[idx]
        filename = self.fnames[idx]

        with h5py.File(seq_path, 'r') as hf:
            dxf_vec = hf['dxf_vec'][0]  # => (S, 46)

            entity_type = dxf_vec[:, 0].astype(np.int64)
            entity_param = dxf_vec[:, 1:].astype(np.float32)

        if entity_type.shape[0] > self.max_nodes:
            entity_type = entity_type[:self.max_nodes]
            entity_param = entity_param[:self.max_nodes, :]

        e_type_t = torch.tensor(entity_type, dtype=torch.long)
        e_param_t = torch.tensor(entity_param, dtype=torch.float32)

        return geom_matrices, (e_type_t, e_param_t), filename


def collate_fn(batch):

    feat_list, adj_list, mask_list, pos2d_list = [], [], [], []
    etype_list, eparam_list, fnames = [], [], []

    for (feat, adj, mask, pos2d), (etype, eparam), fname in batch:
        feat_list.append(torch.tensor(feat, dtype=torch.float32))
        adj_list.append(torch.tensor(adj, dtype=torch.float32))
        mask_list.append(torch.tensor(mask, dtype=torch.float32))
        pos2d_list.append(torch.tensor(pos2d, dtype=torch.float32))
        etype_list.append(etype)
        eparam_list.append(eparam)
        fnames.append(fname)

    feats = torch.stack(feat_list, dim=0)   # [B, max_n, 46]
    adjs  = torch.stack(adj_list, dim=0)    # [B, max_n, max_n]
    masks = torch.stack(mask_list, dim=0)   # [B, max_n]
    pos2d = torch.stack(pos2d_list, dim=0)  # [B, max_n, 2]

    etype_tensor  = torch.stack(etype_list, dim=0)   # [B, seq_len]
    eparam_tensor = torch.stack(eparam_list, dim=0)  # [B, seq_len, 45]

    return (feats, adjs, masks, pos2d), (etype_tensor, eparam_tensor), fnames

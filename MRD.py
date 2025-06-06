#MRD.py
import os
import json
import h5py
import torch
import argparse
import numpy as np
from Sim import get_config, EarlyStopping
from Sim import GeomModel, GeomNodeAlign
from Sim import SeqTransformer, SeqNodeAlign, SelfAttPool
from Sim import encode_dxf
import torch.nn as nn


def compute_recall_at_n(retrieved_list, relevant_set, top_n):
    """Recall@N = (number of retrieved positives) / (total number of positives)"""
    retrieved_top = retrieved_list[:top_n]
    hits = sum(1 for item in retrieved_top if item in relevant_set)
    if len(relevant_set) == 0:
        return 0.0
    recall = hits / len(relevant_set)
    return recall


def compute_ap_at_n(ranked_list, relevant_set, top_n):
    """
    Calculate AP@N (Average Precision),
    only considering documents within top_n range.
    For ranking list ranked_list[0] is the most similar file.
    """
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(ranked_list[:top_n], start=1):
        if item in relevant_set:
            hits += 1
            sum_precisions += hits / i
    if hits == 0:
        return 0.0
    return sum_precisions / hits


def compute_ndcg_at_n(ranked_list, relevant_set, top_n):
    """
    Calculate NDCG@N (Normalized Discounted Cumulative Gain).
    In binary relevance (relevant/irrelevant) scenario,
      DCG = \sum_{i=1}^N (rel_i / log2(i+1)),
      where rel_i=1 if the position is a relevant document, otherwise 0.
    IDCG = DCG when relevant documents are ranked at the top in ideal case.
    """
    def dcg_at_n(r_list, r_set, n):
        dcg = 0.0
        for i, item in enumerate(r_list[:n], start=1):
            if item in r_set:
                dcg += 1.0 / np.log2(i + 1)
        return dcg

    actual_dcg = dcg_at_n(ranked_list, relevant_set, top_n)
    ideal_rank = [1]*min(len(relevant_set), top_n)
    ideal_dcg = 0.0
    for i, rel in enumerate(ideal_rank, start=1):
        ideal_dcg += rel / np.log2(i + 1)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def build_file_groups_from_txt(txt_path):

    with open(txt_path, 'r', encoding='utf-8') as f:
        group_list = json.load(f)

    file2positives = {}
    for group in group_list:
        group_noext = [os.path.splitext(item)[0] for item in group]

        for item in group_noext:
            if item not in file2positives:
                file2positives[item] = set()
            for other in group_noext:
                if other != item:
                    file2positives[item].add(other)

    return file2positives


def main():
    parser = argparse.ArgumentParser("Compute AP/Recall/NDCG on all matched .json/.h5 pairs.")
    parser.add_argument("--geom_dir", type=str, default="/home/vllm/DualDXF/data/Geom/SuperLFD_evaluate",
                        help="Directory containing JSON files")
    parser.add_argument("--seq_dir",  type=str, default="/home/vllm/DualDXF/data/Seq/SuperLFD_evaluate",
                        help="Directory containing H5 files")
    parser.add_argument("--model_ckpt", type=str, default="/home/vllm/DualDXF/checkpoints/Dual_best.pth",
                        help="Path to trained MC model weight file")
    parser.add_argument("--txt_path",   type=str, default="/home/vllm/DualDXF/data/group/super_scale_evaluate.txt",
                        help="txt file containing file groups for determining positive files")
    parser.add_argument("--gpu_id", type=int, default=0, help="Which GPU to use, -1 means CPU")
    args = parser.parse_args()

    geom_files = [f for f in os.listdir(args.geom_dir) if f.endswith(".json")]
    seq_files  = [f for f in os.listdir(args.seq_dir)  if f.endswith(".h5")]

    geom_names = set(os.path.splitext(gf)[0] for gf in geom_files)
    seq_names  = set(os.path.splitext(sf)[0] for sf in seq_files)

    common_names = geom_names.intersection(seq_names)
    common_names = sorted(list(common_names))

    print(f"Number of files in geometry directory: {len(geom_files)}")
    print(f"Number of files in sequence directory: {len(seq_files)}")
    print(f"Number of successfully matched files (exist in both): {len(common_names)}")
    if len(common_names) == 0:
        print("No files matched, program exits.")
        return

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu")
    mc_args = get_config()
    mc_args.save_path = args.model_ckpt

    geom_model = GeomModel(
        args=argparse.Namespace(
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
        optimizer= None,
    )

    geom_model.eval()
    seq_model.eval()
    seq_pool.eval()
    Geom_Ref.eval()
    Seq_Ref.eval()

    name2vec = {}

    for name in common_names:
        json_path = os.path.join(args.geom_dir, name + ".json")
        h5_path   = os.path.join(args.seq_dir,  name + ".h5")

        with torch.no_grad():
            dxf_vec = encode_dxf(h5_path, json_path, geom_model, seq_model, seq_pool, Geom_Ref, Seq_Ref, device, mc_args)
            name2vec[name] = dxf_vec.squeeze(0).cpu().numpy()

    all_vectors = [name2vec[n] for n in common_names]
    all_vectors = np.stack(all_vectors, axis=0)
    N = len(common_names)
    sim_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        vec_i = all_vectors[i]
        for j in range(N):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_val = np.dot(vec_i, all_vectors[j])
                sim_matrix[i, j] = sim_val

    file2positives = build_file_groups_from_txt(args.txt_path)

    top_n_values = [10, 20, 30]

    metrics = {}
    for top_n in top_n_values:
        metrics[top_n] = {
            'sum_ap': 0.0,
            'sum_rec': 0.0,
            'sum_ndcg': 0.0,
            'count': 0
        }

    for i, name in enumerate(common_names):
        relevant_set = file2positives.get(name, set())
        relevant_set = relevant_set.intersection(common_names)

        if len(relevant_set) == 0:
            continue

        sims = sim_matrix[i]
        idxs_sorted = np.argsort(-sims)
        idxs_sorted = [idx for idx in idxs_sorted if idx != i]

        retrieved_list = [common_names[idx] for idx in idxs_sorted]

        for top_n in top_n_values:
            rec_val  = compute_recall_at_n(retrieved_list, relevant_set, top_n)
            ap_val   = compute_ap_at_n(retrieved_list, relevant_set, top_n)
            ndcg_val = compute_ndcg_at_n(retrieved_list, relevant_set, top_n)

            metrics[top_n]['sum_rec'] += rec_val
            metrics[top_n]['sum_ap'] += ap_val
            metrics[top_n]['sum_ndcg'] += ndcg_val
            metrics[top_n]['count'] += 1

    if metrics[top_n_values[0]]['count'] == 0:
        print("Warning: All files have no positives found in txt, same group count=0, cannot calculate metrics!")
        return

    print("========================================")
    print(f"Total {metrics[top_n_values[0]]['count']} files have usable positive information in txt")

    for top_n in top_n_values:
        count = metrics[top_n]['count']
        avg_ap = metrics[top_n]['sum_ap'] / count
        avg_rec = metrics[top_n]['sum_rec'] / count
        avg_ndcg = metrics[top_n]['sum_ndcg'] / count

        print(f"\n--- Evaluation results for top_n = {top_n} ---")
        print(f"AP@{top_n}     = {avg_ap:.4f}")
        print(f"Recall@{top_n} = {avg_rec:.4f}")
        print(f"NDCG@{top_n}   = {avg_ndcg:.4f}")

    print("\n========================================")


if __name__ == "__main__":
    main()
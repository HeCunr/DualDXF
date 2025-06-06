# train.py
import math
import torch
import os
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from config.config import get_config
import argparse
from dataset.dataset import Dataset, collate_fn
from utils.early_stopping import EarlyStopping
from model.GeomLayers.GeomModel import GeomModel
from model.GeomLayers.GeomAlignment import NodeAlignmentHead as GeomNodeAlign
from model.SeqLayers.seq_transformer_encoder import SeqTransformer
from model.SeqLayers.SelfAttPool import SelfAttPool
from model.SeqLayers.SeqAlignment import NodeAlignmentHead as SeqNodeAlign
from model.GeomLayers.Geom_extended_loss import GeomExtendedContrastiveLoss
from model.SeqLayers.Seq_extended_loss import SeqExtendedContrastiveLoss
from model.GeomSeqLoss import GeomSeqLoss
from model.SeqLayers.Seq_loss import SeqContrastiveLoss
from utils.Geom_augment import drop_feature, aug_random_edge, drop_pos2d
from utils.Seq_augment import augment_seq_sample

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def linear_warmup_cosine_decay(epoch, total_epochs, warmup_epochs, min_lr_factor=0.1):
    if epoch < warmup_epochs:
        return float(epoch) / float(max(1, warmup_epochs))
    else:
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cos_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = min_lr_factor + (1.0 - min_lr_factor) * cos_scale
        return scale

def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0):

    warmup_schedule = np.ones(epochs)
    if warmup_epochs > 0:
        warmup_schedule[:warmup_epochs] = np.linspace(start_warmup_value, base_value, warmup_epochs)

    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * np.arange(epochs) / epochs))
    schedule[0:warmup_epochs] = warmup_schedule[:warmup_epochs]

    return schedule

def batch_augment_seq(e_type, e_param, device):

    out_type_list, out_param_list = [], []
    B_ = e_type.size(0)
    max_len = e_type.size(1)

    for i in range(B_):
        arr_type = e_type[i].cpu().numpy()
        arr_param = e_param[i].cpu().numpy()

        combined_arr = np.zeros((max_len, 46), dtype=np.float32)
        combined_arr[:, 0] = arr_type
        combined_arr[:, 1:] = arr_param

        augmented_arr = augment_seq_sample(combined_arr)

        out_type_list.append(augmented_arr[:, 0])
        out_param_list.append(augmented_arr[:, 1:])

    out_type = torch.tensor(np.stack(out_type_list), dtype=torch.long, device=device)
    out_param = torch.tensor(np.stack(out_param_list), dtype=torch.float32, device=device)
    return out_type, out_param


def train_epoch(
        geom_model, geom_align,
        seq_model, seq_align, seq_pool,
        seq_contrast_loss, geom_seq_loss_fn,
        Geom_Ref, Seq_Ref,
        dataloader, optimizer, device, args,
        geom_ext_loss_fn=None,
        seq_ext_loss_fn=None,
        gs_weight=None
):
    geom_model.train()
    geom_align.train()
    seq_model.train()
    seq_align.train()
    seq_pool.train()
    Geom_Ref.train()
    Seq_Ref.train()

    total_loss = 0.0
    total_gg_loss = 0.0
    total_ss_loss = 0.0
    total_gs_loss = 0.0

    for (geom_feats, geom_adjs, geom_masks, geom_pos2d), (seq_etype, seq_eparam), _ in tqdm(dataloader, desc="Train", leave=False):
        geom_feats = geom_feats.to(device)
        geom_adjs  = geom_adjs.to(device)
        geom_masks = geom_masks.to(device)
        geom_pos2d = geom_pos2d.to(device)

        seq_etype  = seq_etype.to(device)
        seq_eparam = seq_eparam.to(device)

        g1_feat = drop_feature(geom_feats.clone(), 0.3)
        g2_feat = drop_feature(geom_feats.clone(), 0.3)
        g1_adj  = aug_random_edge(geom_adjs.clone(), 0.3)
        g2_adj  = aug_random_edge(geom_adjs.clone(), 0.3)
        g1_pos  = drop_pos2d(geom_pos2d.clone(), 0.3)
        g2_pos  = drop_pos2d(geom_pos2d.clone(), 0.3)

        g1_final, _, g1_adj_enc, g1_mask_enc = geom_model(g1_feat, g1_pos, g1_adj, geom_masks)
        g2_final, _, g2_adj_enc, g2_mask_enc = geom_model(g2_feat, g2_pos, g2_adj, geom_masks)

        if args.use_alignment_for_GG:
            g1_aligned, g2_aligned = geom_align.perform_alignment(g1_final, g2_final)
            geom_inputs = (g1_aligned, g2_aligned)
        else:
            geom_inputs = (g1_final, g2_final)

        if not args.Geom_use_extended_loss:
            L_GG = geom_align.loss(*geom_inputs)
        else:
            L_GG = geom_ext_loss_fn(
                *geom_inputs,
                g1_adj_enc, g2_adj_enc,
                g1_mask_enc, g2_mask_enc
            )

        s1_type, s1_param = batch_augment_seq(seq_etype, seq_eparam, device)
        s2_type, s2_param = batch_augment_seq(seq_etype, seq_eparam, device)

        s1_enc = seq_model(s1_type, s1_param)
        s2_enc = seq_model(s2_type, s2_param)

        if args.use_alignment_for_SS:
            s1_aligned, s2_aligned = seq_align.perform_alignment(s1_enc, s2_enc)
            seq_inputs = {"proj_z1": s1_aligned, "proj_z2": s2_aligned}
        else:
            seq_inputs = {"proj_z1": s1_enc, "proj_z2": s2_enc}

        if not args.Seq_use_extended_loss:
            seq_loss_dict = seq_contrast_loss(seq_inputs)
            L_SS = seq_loss_dict["loss_contrastive"]
        else:
            seq_loss_dict = seq_ext_loss_fn(seq_inputs)
            L_SS = seq_loss_dict["loss_contrastive"]

        g_enc_final, g_enc_stim, _, _ = geom_model(geom_feats, geom_pos2d, geom_adjs, geom_masks)
        g_enc_repr = Geom_Ref(g_enc_stim)
        g_enc_repr = nn.functional.normalize(g_enc_repr, dim=1)

        s_enc_clean = seq_model(seq_etype, seq_eparam)
        s_stim = seq_pool(s_enc_clean)
        s_repr = Seq_Ref(s_stim)
        s_repr = nn.functional.normalize(s_repr, dim=1)

        L_GS = geom_seq_loss_fn(g_enc_repr, s_repr)

        weighted_L_GS = gs_weight * args.lambda3 * L_GS
        loss = args.lambda1 * L_GG + args.lambda2 * L_SS + weighted_L_GS

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_gg_loss += L_GG.item() * args.lambda1
        total_ss_loss += L_SS.item() * args.lambda2
        total_gs_loss += L_GS.item() * args.lambda3 * gs_weight

    avg_loss = total_loss / len(dataloader)
    avg_gg_loss = total_gg_loss / len(dataloader)
    avg_ss_loss = total_ss_loss / len(dataloader)
    avg_gs_loss = total_gs_loss / len(dataloader)

    return avg_loss, avg_gg_loss, avg_ss_loss, avg_gs_loss


@torch.no_grad()
def eval_epoch(
        geom_model, geom_align,
        seq_model, seq_align, seq_pool,
        seq_contrast_loss, geom_seq_loss_fn,
        Geom_Ref, Seq_Ref,
        dataloader, device, args,
        geom_ext_loss_fn=None,
        seq_ext_loss_fn=None,
        gs_weight=None
):
    geom_model.eval()
    geom_align.eval()
    seq_model.eval()
    seq_align.eval()
    seq_pool.eval()
    Geom_Ref.eval()
    Seq_Ref.eval()

    total_loss = 0.0
    total_gg_loss = 0.0
    total_ss_loss = 0.0
    total_gs_loss = 0.0

    for (geom_feats, geom_adjs, geom_masks, geom_pos2d), (seq_etype, seq_eparam), _ in tqdm(dataloader, desc="Eval", leave=False):
        geom_feats = geom_feats.to(device)
        geom_adjs  = geom_adjs.to(device)
        geom_masks = geom_masks.to(device)
        geom_pos2d = geom_pos2d.to(device)

        seq_etype  = seq_etype.to(device)
        seq_eparam = seq_eparam.to(device)

        g1_feat = drop_feature(geom_feats.clone(), 0.3)
        g2_feat = drop_feature(geom_feats.clone(), 0.3)
        g1_adj  = aug_random_edge(geom_adjs.clone(), 0.3)
        g2_adj  = aug_random_edge(geom_adjs.clone(), 0.3)
        g1_pos  = drop_pos2d(geom_pos2d.clone(), 0.3)
        g2_pos  = drop_pos2d(geom_pos2d.clone(), 0.3)

        g1_final, _, g1_adj_enc, g1_mask_enc = geom_model(g1_feat, g1_pos, g1_adj, geom_masks)
        g2_final, _, g2_adj_enc, g2_mask_enc = geom_model(g2_feat, g2_pos, g2_adj, geom_masks)

        if args.use_alignment_for_GG:
            g1_aligned, g2_aligned = geom_align.perform_alignment(g1_final, g2_final)
            geom_inputs = (g1_aligned, g2_aligned)
        else:
            geom_inputs = (g1_final, g2_final)

        if not args.Geom_use_extended_loss:
            L_GG = geom_align.loss(*geom_inputs)
        else:
            L_GG = geom_ext_loss_fn(
                *geom_inputs,
                g1_adj_enc, g2_adj_enc,
                g1_mask_enc, g2_mask_enc
            )

        s1_type, s1_param = batch_augment_seq(seq_etype, seq_eparam, device)
        s2_type, s2_param = batch_augment_seq(seq_etype, seq_eparam, device)

        s1_enc = seq_model(s1_type, s1_param)
        s2_enc = seq_model(s2_type, s2_param)

        if args.use_alignment_for_SS:
            s1_aligned, s2_aligned = seq_align.perform_alignment(s1_enc, s2_enc)
            seq_inputs = {"proj_z1": s1_aligned, "proj_z2": s2_aligned}
        else:
            seq_inputs = {"proj_z1": s1_enc, "proj_z2": s2_enc}

        if not args.Seq_use_extended_loss:
            seq_loss_dict = seq_contrast_loss(seq_inputs)
            L_SS = seq_loss_dict["loss_contrastive"]
        else:
            seq_loss_dict = seq_ext_loss_fn(seq_inputs)
            L_SS = seq_loss_dict["loss_contrastive"]

        g_enc_final, g_enc_stim, _, _ = geom_model(geom_feats, geom_pos2d, geom_adjs, geom_masks)
        g_enc_repr = Geom_Ref(g_enc_stim)
        g_enc_repr = nn.functional.normalize(g_enc_repr, dim=1)

        s_enc_clean = seq_model(seq_etype, seq_eparam)
        s_stim = seq_pool(s_enc_clean)
        s_repr = Seq_Ref(s_stim)
        s_repr = nn.functional.normalize(s_repr, dim=1)

        L_GS = geom_seq_loss_fn(g_enc_repr, s_repr)

        weighted_L_GS = gs_weight * args.lambda3 * L_GS
        loss = args.lambda1 * L_GG + args.lambda2 * L_SS + weighted_L_GS

        total_loss += loss.item()
        total_gg_loss += L_GG.item() * args.lambda1
        total_ss_loss += L_SS.item() * args.lambda2
        total_gs_loss += L_GS.item() * args.lambda3 * gs_weight

    avg_loss = total_loss / len(dataloader)
    avg_gg_loss = total_gg_loss / len(dataloader)
    avg_ss_loss = total_ss_loss / len(dataloader)
    avg_gs_loss = total_gs_loss / len(dataloader)

    return avg_loss, avg_gg_loss, avg_ss_loss, avg_gs_loss

def main():
    args = get_config()
    set_seed(args.seed)
    device = torch.device(
        f'cuda:{args.gpu_id}' if (torch.cuda.is_available() and args.gpu_id >= 0) else 'cpu'
    )

    if args.wandb_on:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args)
        )

    dataset = Dataset(args.geom_data_dir, args.seq_data_dir, max_nodes=args.max_n)
    total_size = len(dataset)
    train_sz = int(total_size * args.train_ratio)
    val_sz   = int(total_size * args.val_ratio)
    test_sz  = total_size - train_sz - val_sz

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_sz, val_sz, test_sz],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    geom_model = GeomModel(
        args = argparse.Namespace(
            filters=args.geom_filters,
            conv=args.geom_conv_type,
            dropout=args.geom_dropout,
            graph_init_dim=46
        ),
        d_model=args.d_model
    ).to(device)

    geom_align = GeomNodeAlign(
        d_model=args.d_model,
        alignment='concat',
        perspectives=args.perspectives,
        tau=0.07
    ).to(device)

    seq_model = SeqTransformer(
        d_model=args.seq_d_model,
        num_layers=args.seq_num_layers,
        nhead=args.seq_nhead,
        dim_feedforward=args.seq_dim_feedforward,
        dropout=args.seq_dropout,
        max_len=args.max_len,
        input_length=args.input_length,
        output_length=args.output_length
    ).to(device)

    seq_align = SeqNodeAlign(
        d_model=args.d_model,
        alignment='concat',
        latent_dropout=0.1
    ).to(device)

    seq_pool = SelfAttPool(d_model=args.d_model).to(device)

    class DummySeqCfg:
        loss_weights = {"loss_cl_weight": 1.0}

    seq_contrast_loss = SeqContrastiveLoss(
        cfg=DummySeqCfg(),
        device=device,
        batch_size=args.batch_size,
        temperature=0.07
    ).to(device)

    seq_ext_loss_fn = SeqExtendedContrastiveLoss(
        cfg=argparse.Namespace(
            loss_weights = {"loss_cl_weight": 1.0},
            extend_top_k = args.seq_ext_top_k,
            alpha        = args.seq_ext_alpha,
            scales       = args.seq_ext_scales,
            local_k      = args.seq_ext_local_k,
            sigma        = args.seq_ext_sigma
        ),
        device      = device,
        batch_size  = args.batch_size,
        temperature = args.seq_ext_temperature
    ).to(device)

    geom_seq_loss_fn = GeomSeqLoss(temperature=0.07).to(device)

    geom_ext_loss_fn = GeomExtendedContrastiveLoss(
        alpha=args.geom_ext_alpha,
        top_k=args.geom_ext_top_k,
        scales_graph=args.geom_ext_scales_graph,
        scales_feat=args.geom_ext_scales_feat,
        tau=args.geom_ext_tau
    ).to(device)

    Geom_Ref = nn.Sequential(
        nn.Linear(args.d_model, args.d_model),
        nn.ReLU(),
        nn.Linear(args.d_model, args.d_model)
    ).to(device)
    Seq_Ref = nn.Sequential(
        nn.Linear(args.d_model, args.d_model),
        nn.ReLU(),
        nn.Linear(args.d_model, args.d_model)
    ).to(device)

    all_params = (
            list(geom_model.parameters())
            + list(geom_align.parameters())
            + list(seq_model.parameters())
            + list(seq_align.parameters())
            + list(seq_pool.parameters())
            + list(Geom_Ref.parameters())
            + list(Seq_Ref.parameters())
    )

    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    total_epochs = args.epochs
    warmup_epochs = int(total_epochs * args.warmup_ratio)
    min_lr_factor = args.min_lr_factor

    def lr_lambda(epoch):
        return linear_warmup_cosine_decay(epoch, total_epochs, warmup_epochs, min_lr_factor)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    early_stopper = EarlyStopping(patience=args.patience, checkpoint_path=args.save_path)

    if args.wandb_on:
        wandb.watch(
            [geom_model, geom_align, seq_model, seq_align, seq_pool, Geom_Ref, Seq_Ref],
            log="all"
        )

    phase1_end = int(total_epochs * args.phase1_ratio)
    transition_end = phase1_end + int(total_epochs * args.phase_transition_ratio)
    gs_weight_schedule = np.zeros(total_epochs)
    if phase1_end < total_epochs:
        gs_weight_schedule[:phase1_end] = args.gs_weight_phase1

        if transition_end > phase1_end:
            transition_epochs = transition_end - phase1_end
            weight_diff = args.gs_weight_phase2 - args.gs_weight_phase1
            transition_weights = args.gs_weight_phase1 + weight_diff * 0.5 * (1 - np.cos(np.pi * np.arange(transition_epochs) / transition_epochs))
            gs_weight_schedule[phase1_end:transition_end] = transition_weights

        if transition_end < total_epochs:
            gs_weight_schedule[transition_end:] = args.gs_weight_phase2

    print(f"Single-modal pre-training: Epochs 0-{phase1_end-1}, GS={args.gs_weight_phase1}")
    print(f"smooth transition: Epochs {phase1_end}-{transition_end-1}, GS={args.gs_weight_phase1} to {args.gs_weight_phase2}")
    print(f"cross-modal fine-tuning: Epochs {transition_end}-{total_epochs-1}, GS={args.gs_weight_phase2}")

    for epoch in range(total_epochs):
        current_gs_weight = gs_weight_schedule[epoch]

        if epoch < phase1_end:
            current_phase = 'phase1'
        elif epoch < transition_end:
            current_phase = 'transition'
        else:
            current_phase = 'phase2'

        if epoch == 0:
            print(f"Starting phase 1: Unimodal pre-training (L_GG + L_SS)")
        elif epoch == phase1_end:
            print(f"Beginning of transition period: Gradually introducing L_GS")
        elif epoch == transition_end:
            print(f"Start phase 2: Full cross-modal training (L_GG + L_SS + L_GS)")

        train_loss, train_gg_loss, train_ss_loss, train_gs_loss = train_epoch(
            geom_model, geom_align,
            seq_model, seq_align, seq_pool,
            seq_contrast_loss, geom_seq_loss_fn,
            Geom_Ref, Seq_Ref,
            train_loader, optimizer, device, args,
            geom_ext_loss_fn=geom_ext_loss_fn,
            seq_ext_loss_fn=seq_ext_loss_fn,
            gs_weight=current_gs_weight
        )
        scheduler.step()

        val_loss, val_gg_loss, val_ss_loss, val_gs_loss = eval_epoch(
            geom_model, geom_align,
            seq_model, seq_align, seq_pool,
            seq_contrast_loss, geom_seq_loss_fn,
            Geom_Ref, Seq_Ref,
            val_loader, device, args,
            geom_ext_loss_fn=geom_ext_loss_fn,
            seq_ext_loss_fn=seq_ext_loss_fn,
            gs_weight=current_gs_weight
        )

        print(f"[Epoch {epoch+1:03d}/{total_epochs}] [{current_phase}, GS_weight={current_gs_weight:.4f}]")
        print(f"Training: Total loss={train_loss:.4f} (GG={train_gg_loss:.4f}, SS={train_ss_loss:.4f}, GS={train_gs_loss:.4f})")
        print(f"Verification: Total Loss={val_loss:.4f} (GG={val_gg_loss:.4f}, SS={val_ss_loss:.4f}, GS={val_gs_loss:.4f})")
        print(f"Learning Rate={scheduler.get_last_lr()[0]:.6f}")

        if args.wandb_on:
            wandb.log({
                "epoch": epoch,
                "gs_weight": current_gs_weight,
                "train_loss": train_loss,
                "train_gg_loss": train_gg_loss,
                "train_ss_loss": train_ss_loss,
                "train_gs_loss": train_gs_loss,
                "val_loss": val_loss,
                "val_gg_loss": val_gg_loss,
                "val_ss_loss": val_ss_loss,
                "val_gs_loss": val_gs_loss,
                "lr": scheduler.get_last_lr()[0],
                "training_phase": current_phase
            })

        early_stopper(
            val_loss, current_phase,
            geom_model, geom_align,
            seq_model, seq_align, seq_pool,
            Geom_Ref, Seq_Ref,
            optimizer, epoch
        )

        if current_phase == 'phase2' and early_stopper.early_stop:
            print("Early stopping triggered in Phase 2.")
            break

    early_stopper.load_checkpoint(
        geom_model, geom_align,
        seq_model, seq_align, seq_pool,
        Geom_Ref, Seq_Ref,
        optimizer,
        phase='phase2'
    )

    final_test_loss, test_gg_loss, test_ss_loss, test_gs_loss = eval_epoch(
        geom_model, geom_align,
        seq_model, seq_align, seq_pool,
        seq_contrast_loss, geom_seq_loss_fn,
        Geom_Ref, Seq_Ref,
        test_loader, device, args,
        geom_ext_loss_fn=geom_ext_loss_fn,
        seq_ext_loss_fn=seq_ext_loss_fn,
        gs_weight=args.gs_weight_phase2
    )

    print(f"Final Test Metrics:")
    print(f"  Total Loss = {final_test_loss:.4f}")
    print(f"  GG Loss = {test_gg_loss:.4f}")
    print(f"  SS Loss = {test_ss_loss:.4f}")
    print(f"  GS Loss = {test_gs_loss:.4f}")

    if args.wandb_on:
        wandb.log({
            "final_test_loss": final_test_loss,
            "final_test_gg_loss": test_gg_loss,
            "final_test_ss_loss": test_ss_loss,
            "final_test_gs_loss": test_gs_loss
        })
        wandb.finish()

if __name__ == "__main__":
    main()
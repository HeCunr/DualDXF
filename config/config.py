# config/config.py

import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Multi-Contrast Training Config for Geom & Seq")

    parser.add_argument('--geom_data_dir', type=str, default='/home/vllm/DualDXF/data/Geom/SuperLFD_train',
                        help='root directory for the geom dataset (.json)')
    parser.add_argument('--seq_data_dir', type=str, default='/home/vllm/DualDXF/data/Seq/SuperLFD_train',
                        help='root directory for the seq dataset (.h5)')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')

    parser.add_argument('--lambda1', type=float, default=1.0, help='weight for L_GG (Geom-Geom)')
    parser.add_argument('--lambda2', type=float, default=1.0, help='weight for L_SS (Seq-Seq)')
    parser.add_argument('--lambda3', type=float, default=1.5, help='weight for L_GS (Geom-Seq)')

    parser.add_argument('--phase1_ratio', type=float, default=0.4)
    parser.add_argument('--phase_transition_ratio', type=float, default=0.15)
    parser.add_argument('--gs_weight_phase1', type=float, default=0.0)
    parser.add_argument('--gs_weight_phase2', type=float, default=1.0)

    parser.add_argument('--d_model', type=int, default=256)

    parser.add_argument('--Geom_use_extended_loss', action='store_true', default=False,
                        help='Whether to use extended contrastive loss for Geom line')
    parser.add_argument('--Seq_use_extended_loss', action='store_true', default=False,
                        help='Whether to use extended contrastive loss for Seq line')

    parser.add_argument('--geom_ext_alpha', type=float, default=0.5,
                        help='alpha for graph & feature diffusion in GeomExtendedLoss')
    parser.add_argument('--geom_ext_top_k', type=int, default=2,
                        help='top_k for half-positive neighbors in GeomExtendedLoss')
    parser.add_argument('--geom_ext_scales_graph', type=str, default='1,2,3',
                        help='multi-scale graph diffusion steps (e.g. "1,2,3")')
    parser.add_argument('--geom_ext_scales_feat', type=str, default='0.05,0.1,0.2',
                        help='multi-scale feature diffusion temps (e.g. "0.05,0.1,0.2")')
    parser.add_argument('--geom_ext_tau', type=float, default=0.07,
                        help='temperature used in Geom extended contrastive loss')

    parser.add_argument('--seq_ext_top_k', type=int, default=2,
                        help='top_k for half-positive neighbors in SeqExtendedContrastiveLoss')
    parser.add_argument('--seq_ext_alpha', type=float, default=0.5,
                        help='alpha factor for combining positional weight & association in SeqExtendedContrastiveLoss')
    parser.add_argument('--seq_ext_scales', type=str, default='0.05,0.1,0.2',
                        help='multi-scale diffusion temps for extended seq loss')
    parser.add_argument('--seq_ext_local_k', type=int, default=2,
                        help='(optional) local_k param if your extended seq loss uses it')
    parser.add_argument('--seq_ext_sigma', type=float, default=2.0,
                        help='(optional) sigma param for building distance weight in extended seq loss')
    parser.add_argument('--seq_ext_temperature', type=float, default=0.07,
                        help='temperature used in Seq extended contrastive loss')

    parser.add_argument('--use_alignment_for_GG', action='store_true', default=True)
    parser.add_argument('--use_alignment_for_SS', action='store_true', default=True)

    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--save_path', type=str, default='checkpoints/Dual_best.pth', help='model save path')

    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='ratio for training split')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='ratio for validation split')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='ratio for test split')

    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='the ratio of epochs to use for linear warmup')
    parser.add_argument('--min_lr_factor', type=float, default=0.1,
                        help='the factor of base lr as minimum lr in cosine annealing')

    parser.add_argument('--wandb_project', type=str, default="DualDXF",
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default="your_entity_name",
                        help='Weights & Biases entity (username or team name)')
    parser.add_argument('--wandb_run_name', type=str, default="DualDXF_Run",
                        help='W&B run name')
    parser.add_argument('--wandb_log_freq', type=int, default=100,
                        help='Frequency of logging model gradients and parameters')
    parser.add_argument('--wandb_on', action='store_true',default=False,
                        help='Whether to enable wandb logging')

    parser.add_argument('--max_n', type=int, default=4096)
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--input_length', type=int, default=4096)
    parser.add_argument('--output_length', type=int, default=64)

    args = parser.parse_args()

    args.geom_filters = f'{args.d_model}_{args.d_model}_{args.d_model}'
    args.geom_conv_type = 'ggnn'
    args.geom_dropout = 0.1

    args.seq_d_model = args.d_model
    args.seq_num_layers = 6
    args.seq_nhead = 8
    args.seq_dim_feedforward = args.d_model * 2
    args.seq_dropout = 0.1

    args.ref_hidden_dim = args.d_model

    args.perspectives = args.d_model

    args.geom_ext_scales_graph = [int(x) for x in args.geom_ext_scales_graph.split(',')]
    args.geom_ext_scales_feat  = [float(x) for x in args.geom_ext_scales_feat.split(',')]
    args.seq_ext_scales        = [float(x) for x in args.seq_ext_scales.split(',')]

    return args
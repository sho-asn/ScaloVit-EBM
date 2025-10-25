# File: train_ebm_ablation.py
# A flexible training script for the main model and all ablation models.

import os
import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
import numpy as np

# --- Import all model types ---
from ebm_model_vit import EBViTModelWrapper as PatchBasedEBM
from ablation_models import ImageBased_EBViTModelWrapper, ConvHead_EBMWrapper

from ebm_utils import (
    flow_weight, 
    gibbs_sampling_time_sweep, 
    ema, 
    infiniteloop,
    get_warmup_lr_lambda
)

# --- Dependency Check ---
try:
    from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
except ImportError:
    print("You need to install torchcfm: pip install torchcfm")
    exit()

# --- Argument Parsing (with new argument) ---
def get_args():
    parser = argparse.ArgumentParser(description="Train EBM variants for anomaly detection.")
    
    # NEW ARGUMENT to select model type
    parser.add_argument("--ablation_model_type", type=str, default='patch_based', 
                        choices=['patch_based', 'image_based', 'conv_head'], 
                        help="Selects the model architecture to train for ablation.")

    # --- All other arguments are the same as your original script ---
    parser.add_argument("--train_data_path", type=str, default="preprocessed_dataset/train_chunks_wavelet_mag.pt")
    parser.add_argument("--val_data_path", type=str, default="preprocessed_dataset/val_chunks_wavelet_mag.pt")
    parser.add_argument("--output_dir", type=str, default="./results/ebm_training")
    parser.add_argument("--model_name", type=str, default="EBM_ablation_model")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--total_steps", type=int, default=15000)
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_cd", type=float, default=0.001)
    parser.add_argument("--time_cutoff", type=float, default=1.0)
    parser.add_argument("--cd_neg_clamp", type=float, default=0.02)
    parser.add_argument("--cd_trim_fraction", type=float, default=0.1)
    parser.add_argument("--lambda_gp", type=float, default=0.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.0)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--n_gibbs", type=int, default=200)
    parser.add_argument("--dt_gibbs", type=float, default=0.01)
    parser.add_argument("--epsilon_max", type=float, default=0.01)
    parser.add_argument("--split_negative", action="store_true")
    parser.add_argument("--same_temperature_scheduler", action="store_true")
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--channel_mult", type=int, nargs='+', default=[1, 2, 2, 2])
    parser.add_argument("--attention_resolutions", type=str, default="16")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_head_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[128, 8])
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--transformer_nheads", type=int, default=4)
    parser.add_argument("--transformer_nlayers", type=int, default=8)
    parser.add_argument("--output_scale", type=float, default=1000.0)
    parser.add_argument("--energy_clamp", type=float, default=None)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    return parser.parse_args()

# --- MODIFIED Forward Pass to handle different models ---
def forward_all(model, flow_matcher, x_real_flow, x_real_cd, args):
    device = x_real_flow.device
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        x0_flow = torch.randn_like(x_real_flow)
        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0_flow, x_real_flow)
        vt = model(t, xt)
        flow_mse = (vt - ut).square().mean(dim=[1, 2, 3])
        w_flow = flow_weight(t, cutoff=args.time_cutoff)
        loss_flow = torch.mean(w_flow * flow_mse)

        loss_gp = torch.tensor(0.0, device=device)
        if args.lambda_gp > 0.0:
            grad_norm_sq = vt.square().sum(dim=[1, 2, 3])
            loss_gp = args.lambda_gp * grad_norm_sq.mean()

        loss_smooth = torch.tensor(0.0, device=device)
        # MODIFICATION: Only calculate smoothness loss for the patch-based model
        get_tokens_for_smoothness = args.lambda_smooth > 0.0 and args.no_shuffle and (args.ablation_model_type == 'patch_based')

        loss_cd, pos_energy, neg_energy = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        if args.lambda_cd > 0.0 or get_tokens_for_smoothness:
            if get_tokens_for_smoothness:
                pos_energy, pos_embeds = model.potential(x_real_cd, torch.ones_like(t), return_tokens=True)
                chunk_vectors = pos_embeds.mean(dim=1)
                if chunk_vectors.shape[0] > 1:
                    diffs = chunk_vectors[1:] - chunk_vectors[:-1]
                    loss_smooth = args.lambda_smooth * diffs.square().mean()
            else:
                # Ablation models don't support return_tokens, so call it with False
                pos_energy = model.potential(x_real_cd, torch.ones_like(t), return_tokens=False)

            if args.lambda_cd > 0.0:
                # ... (rest of CD logic is the same)
                x_neg_init = torch.randn_like(x_real_cd)
                x_neg = gibbs_sampling_time_sweep(x_init=x_neg_init, model=model, n_steps=args.n_gibbs, dt=args.dt_gibbs, epsilon_max=args.epsilon_max, time_cutoff=args.time_cutoff)
                neg_energy = model.potential(x_neg, torch.ones_like(t))
                cd_val = pos_energy.mean() - neg_energy.mean()
                loss_cd = args.lambda_cd * cd_val
                if args.cd_neg_clamp > 0:
                    loss_cd = torch.maximum(loss_cd, torch.tensor(-args.cd_neg_clamp, device=device))

    total_loss = loss_flow + loss_cd + loss_gp + loss_smooth
    return total_loss, loss_flow, loss_cd, loss_gp, loss_smooth, pos_energy, neg_energy

# --- MODIFIED Training Loop to select model ---
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    print(f"Loading data...")
    train_chunks = torch.load(args.train_data_path)
    if isinstance(train_chunks, list):
        train_chunks = torch.cat(train_chunks, dim=0)
    _, C, H, W = train_chunks.shape
    print(f"Detected data shape: C={C}, H={H}, W={W}")

    train_dataset = TensorDataset(train_chunks, torch.zeros(train_chunks.size(0)))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=not args.no_shuffle, drop_last=True, num_workers=0)
    train_datalooper = infiniteloop(train_loader)

    # MODIFICATION: Select model class based on argument
    if args.ablation_model_type == 'image_based':
        print("--- Training ABLATION model: ImageBased_EBViTModelWrapper ---")
        model_class = ImageBased_EBViTModelWrapper
    elif args.ablation_model_type == 'conv_head':
        print("--- Training ABLATION model: ConvHead_EBMWrapper ---")
        model_class = ConvHead_EBMWrapper
    else: # 'patch_based'
        print("--- Training MAIN model: PatchBasedEBM ---")
        model_class = PatchBasedEBM

    model_args_dict = vars(args)
    model_args_dict['dim'] = (C, H, W)

    # --- Argument Filtering Logic ---
    # 1. Define keys that are for the script/training loop and NOT for any model architecture
    script_and_training_keys = [
        'ablation_model_type', 'train_data_path', 'val_data_path', 'output_dir',
        'model_name', 'resume_ckpt', 'save_step', 'log_step', 'lr', 'batch_size', 
        'total_steps', 'warmup', 'ema_decay', 'grad_clip', 'num_workers', 'use_amp', 
        'gradient_accumulation_steps', 'no_shuffle', 'lambda_cd', 'time_cutoff', 
        'cd_neg_clamp', 'cd_trim_fraction', 'lambda_gp', 'lambda_smooth', 'n_gibbs', 
        'dt_gibbs', 'epsilon_max', 'split_negative', 'same_temperature_scheduler'
    ]
    
    # 2. Start with a dictionary of all potential model args
    clean_model_args = {k: v for k, v in model_args_dict.items() if k not in script_and_training_keys}

    # 3. If the model is the conv_head, it does not use any ViT arguments, so remove them.
    if args.ablation_model_type == 'conv_head':
        vit_specific_keys = ['patch_size', 'embed_dim', 'transformer_nheads', 'transformer_nlayers', 'include_pos_embed']
        for key in vit_specific_keys:
            clean_model_args.pop(key, None)

    model = model_class(**clean_model_args).to(device)

    ema_model = copy.deepcopy(model).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_warmup_lr_lambda(args.warmup))
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    start_step = 0
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        checkpoint = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_step = checkpoint['step'] + 1

    print("Starting training...")
    last_log_time = time.time()
    optimizer.zero_grad()

    for step in range(start_step, args.total_steps):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            x_real_flow, _ = next(train_datalooper)
            x_real_cd, _ = next(train_datalooper)
            x_real_flow = x_real_flow.to(device).float()
            x_real_cd = x_real_cd.to(device).float()
            total_loss, loss_flow, loss_cd, loss_gp, loss_smooth, pos_energy, neg_energy = forward_all(model, flow_matcher, x_real_flow, x_real_cd, args)
            loss = total_loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            ema(model, ema_model, args.ema_decay)

        if step % args.log_step == 0:
            now = time.time()
            elapsed = now - last_log_time
            sps = args.log_step / elapsed if elapsed > 0 else 0
            last_log_time = now
            curr_lr = scheduler.get_last_lr()[0]
            print(
                f"[Step {step}/{args.total_steps}] "
                f"Loss: {total_loss.item():.4f} (Flow: {loss_flow.item():.4f}, CD: {loss_cd.item():.4f}, GP: {loss_gp.item():.4f}, Smooth: {loss_smooth.item():.4f}) | "
                f"Pos E: {pos_energy.mean().item():.2f} (std: {pos_energy.std().item():.2f}) | "
                f"Neg E: {neg_energy.mean().item():.2f} (std: {neg_energy.std().item():.2f}) | "
                f"LR: {curr_lr:.6f} | {sps:.2f} it/s"
            )

        if step > 0 and step % args.save_step == 0:
            ckpt_path = os.path.join(args.output_dir, f"{args.model_name}_step_{step}.pt")
            torch.save({'model': model.state_dict(), 'ema_model': ema_model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': step, 'args': args}, ckpt_path)
            print(f"--- Checkpoint saved to {ckpt_path} ---")

    print("Training finished.")
    final_ckpt_path = os.path.join(args.output_dir, f"{args.model_name}_final.pt")
    torch.save({'model': model.state_dict(), 'ema_model': ema_model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': args.total_steps - 1, 'args': args}, final_ckpt_path)
    print(f"--- Final checkpoint saved to {final_ckpt_path} ---")

if __name__ == "__main__":
    args = get_args()
    train(args)

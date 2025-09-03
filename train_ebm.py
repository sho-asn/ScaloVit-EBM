import os
import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
import numpy as np
import deepspeed

# --- Local Imports ---
from ebm_model_vit import EBViTModelWrapper as EBM
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

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Train an Energy-Based Model for Anomaly Detection using ViT architecture")
    
    # --- Add DeepSpeed-specific arguments ---
    parser = deepspeed.add_config_arguments(parser)

    # Data and Paths
    parser.add_argument("--train_data_path", type=str, default="preprocessed_dataset/train_chunks_wavelet.pt", help="Path to the training data chunks file.")
    parser.add_argument("--val_data_path", type=str, default="preprocessed_dataset/val_chunks_wavelet.pt", help="Path to the validation data chunks file.")
    parser.add_argument("--output_dir", type=str, default="./results/ebm_training", help="Directory for results and checkpoints.")
    parser.add_argument("--model_name", type=str, default="EBM_scalogram_vit", help="Name for the model and output files.")
    
    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    # batch_size is handled by deepspeed config
    parser.add_argument("--total_steps", type=int, default=15000, help="Total training steps.")
    parser.add_argument("--warmup", type=int, default=10000, help="Learning rate warmup steps.")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate.")
    # grad_clip is handled by deepspeed config
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")

    # Loss Configuration
    parser.add_argument("--lambda_cd", type=float, default=0.001, help="Coefficient for Contrastive Divergence loss.")
    parser.add_argument("--time_cutoff", type=float, default=1.0, help="Flow loss decays to zero beyond t>=time_cutoff")
    parser.add_argument("--cd_neg_clamp", type=float, default=0.02, help="Clamp negative total CD below -cd_neg_clamp. 0=disable clamp.")
    parser.add_argument("--cd_trim_fraction", type=float, default=0.1, help="Fraction of highest negative energies discarded for CD (0=disable).")

    # Gibbs Sampling for CD
    parser.add_argument("--n_gibbs", type=int, default=200, help="Number of Gibbs steps for CD.")
    parser.add_argument("--dt_gibbs", type=float, default=0.01, help="Step size for Gibbs sampling.")
    parser.add_argument("--epsilon_max", type=float, default=0.01, help="Max step size in Gibbs sampling temperature schedule.")
    parser.add_argument("--split_negative", action="store_true", help="If set, initialize half of the negative samples from x_real_cd, half from noise")
    parser.add_argument("--same_temperature_scheduler", action="store_true", help="If set, use the same temperature schedule for all samples in CD.")

    # Model Architecture
    parser.add_argument("--num_channels", type=int, default=128, help="Base channels for the U-Net.")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per stage.")
    parser.add_argument("--channel_mult", type=int, nargs='+', default=[1, 2, 4], help="Channel multipliers for each U-Net resolution block.")
    parser.add_argument("--attention_resolutions", type=str, default="16", help="Resolutions for U-Net self-attention.")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads for U-Net.")
    parser.add_argument("--num_head_channels", type=int, default=64, help="Number of channels per U-Net attention head.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the ViT head.")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimension for ViT head.")
    parser.add_argument("--transformer_nheads", type=int, default=4, help="Number of heads in the ViT encoder.")
    parser.add_argument("--transformer_nlayers", type=int, default=8, help="Number of layers in the ViT encoder.")
    parser.add_argument("--output_scale", type=float, default=1000.0, help="Multiplier for the final energy output.")
    parser.add_argument("--energy_clamp", type=float, default=None, help="Tanh-based clamp for energy.")

    # Checkpoint & Logging
    parser.add_argument("--save_step", type=int, default=1000, help="Checkpoint save and validation frequency.")
    parser.add_argument("--log_step", type=int, default=100, help="Logging frequency.")

    return parser.parse_args()

# --- Forward Pass --- 
def forward_all(model, flow_matcher, x_real_flow, x_real_cd, args):
    # device is handled by deepspeed
    # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    # 1) Flow matching loss
    x0_flow = torch.randn_like(x_real_flow)
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0_flow, x_real_flow)
    vt = model(t, xt)
    flow_mse = (vt - ut).square().mean(dim=[1, 2, 3])
    w_flow = flow_weight(t, cutoff=args.time_cutoff)
    loss_flow = torch.mean(w_flow * flow_mse)

    # 2) Contrastive Divergence loss
    loss_cd = torch.tensor(0.0, device=model.device)
    pos_energy = torch.tensor(0.0, device=model.device)
    neg_energy = torch.tensor(0.0, device=model.device)

    if args.lambda_cd > 0.0:
        pos_energy = model.potential(x_real_cd, torch.ones_like(t))

        if args.split_negative:
            B = x_real_cd.size(0)
            half_b = B // 2
            x_neg_init = torch.empty_like(x_real_cd)
            x_neg_init[:half_b] = x_real_cd[:half_b]
            x_neg_init[half_b:] = torch.randn_like(x_neg_init[half_b:])
            at_data_mask = torch.zeros(B, dtype=torch.bool, device=model.device)
            at_data_mask[:half_b] = True
        else:
            x_neg_init = torch.randn_like(x_real_cd)
            at_data_mask = torch.zeros(x_real_cd.size(0), dtype=torch.bool, device=model.device)

        if args.same_temperature_scheduler:
            at_data_mask.fill_(False)

        x_neg = gibbs_sampling_time_sweep(
            x_init=x_neg_init,
            model=model,
            at_data_mask=at_data_mask,
            n_steps=args.n_gibbs,
            dt=args.dt_gibbs,
            epsilon_max=args.epsilon_max,
            time_cutoff=args.time_cutoff
        )
        neg_energy = model.potential(x_neg, torch.ones_like(t))

        if args.cd_trim_fraction > 0.0:
            total_patch_energies = neg_energy.numel()
            k = int(args.cd_trim_fraction * total_patch_energies)
            if k > 0:
                neg_sorted, _ = neg_energy.view(-1).sort()
                neg_trimmed = neg_sorted[:-k]
                neg_stat = neg_trimmed.mean()
            else:
                neg_stat = neg_energy.mean()
        else:
            neg_stat = neg_energy.mean()

        cd_val = pos_energy.mean() - neg_stat
        loss_cd = args.lambda_cd * cd_val

        if args.cd_neg_clamp > 0:
            loss_cd = torch.maximum(loss_cd, torch.tensor(-args.cd_neg_clamp, device=model.device))

    total_loss = loss_flow + loss_cd
    return total_loss, loss_flow, loss_cd, pos_energy, neg_energy

# --- Training Loop ---
def train(args):
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data
    print("Loading data...")
    train_chunks = torch.load(args.train_data_path)
    if isinstance(train_chunks, list):
        train_chunks = torch.cat(train_chunks, dim=0)
    
    _, C, H, W = train_chunks.shape
    print(f"Detected data shape: C={C}, H={H}, W={W}")
    train_dataset = TensorDataset(train_chunks, torch.zeros(train_chunks.size(0)))

    # Model
    model = EBM(
        dim=(C, H, W),
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        dropout=args.dropout,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        transformer_nheads=args.transformer_nheads,
        transformer_nlayers=args.transformer_nlayers,
        output_scale=args.output_scale,
        energy_clamp=args.energy_clamp,
    )

    ema_model = copy.deepcopy(model)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # --- DeepSpeed Initialization ---
    # Optimizer and scheduler are defined in ds_config.json or passed as args.
    # We pass the model parameters to DeepSpeed.
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset
    )
    
    # We need a separate datalooper for the contrastive divergence part
    cd_datalooper = infiniteloop(train_loader)

    print(f"Using device: {model_engine.device}")

    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    # Resume from checkpoint if provided
    # Note: DeepSpeed handles loading the model/optimizer/scheduler states.
    start_step = 0
    if args.resume_ckpt:
        # The 'resume_ckpt' should be a directory for DeepSpeed
        _, client_sd = model_engine.load_checkpoint(args.resume_ckpt)
        if client_sd is not None:
            start_step = client_sd.get('step', 0)
            print(f"Resuming from step {start_step}")

    # Training
    print("Starting training...")
    last_log_time = time.time()
    for step, (x_real_flow, _) in enumerate(train_loader, start=start_step):
        if step >= args.total_steps:
            break

        x_real_cd, _ = next(cd_datalooper)
        
        # Data is automatically moved to the correct device by DeepSpeed loader

        total_loss, loss_flow, loss_cd, pos_energy, neg_energy = forward_all(
            model_engine, flow_matcher, x_real_flow, x_real_cd, args
        )

        model_engine.backward(total_loss)
        model_engine.step()
        
        # EMA update needs to be done on the unwrapped model
        ema(model_engine.module, ema_model, args.ema_decay)

        if step % args.log_step == 0:
            now = time.time()
            elapsed = now - last_log_time
            sps = args.log_step / elapsed if elapsed > 0 else 0
            last_log_time = now
            # LR is handled by DeepSpeed scheduler, can be logged if needed
            print(
                f"[Step {step}/{args.total_steps}] "
                f"Loss: {total_loss.item():.4f} (Flow: {loss_flow.item():.4f}, CD: {loss_cd.item():.4f}) | "
                f"Pos E: {pos_energy.mean().item():.2f} (std: {pos_energy.std().item():.2f}) | "
                f"Neg E: {neg_energy.mean().item():.2f} (std: {neg_energy.std().item():.2f}) | "
                f"{sps:.2f} it/s"
            )

        if step > 0 and step % args.save_step == 0:
            # DeepSpeed checkpointing
            checkpoint_dir = os.path.join(args.output_dir, str(step))
            client_sd = {'step': step, 'args': args, 'ema_model': ema_model.state_dict()}
            model_engine.save_checkpoint(checkpoint_dir, client_state=client_sd)
            print(f"--- Checkpoint saved to {checkpoint_dir} ---")

    print("Training finished.")

    # --- Save Final Model ---
    print("Saving final model...")
    final_checkpoint_dir = os.path.join(args.output_dir, "final")
    client_sd = {'step': args.total_steps - 1, 'args': args, 'ema_model': ema_model.state_dict()}
    model_engine.save_checkpoint(final_checkpoint_dir, client_state=client_sd)
    print(f"--- Final checkpoint saved to {final_checkpoint_dir} ---")

if __name__ == "__main__":
    args = get_args()
    train(args)
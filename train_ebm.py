import os
import time
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse
import numpy as np

# --- Local Imports ---
from ebm_model import EBM
from ebm_utils import flow_weight, gibbs_sampling_time_sweep, ema, infiniteloop

# --- Dependency Check ---
try:
    from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
except ImportError:
    print("You need to install torchcfm: pip install torchcfm")
    exit()

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Train an Energy-Based Model for Anomaly Detection")
    parser.add_argument("--train_data_path", type=str, default="preprocessed_dataset/train_chunks.pt", help="Path to the training data chunks file.")
    parser.add_argument("--val_data_path", type=str, default="preprocessed_dataset/val_chunks.pt", help="Path to the validation data chunks file.")
    parser.add_argument("--output_dir", type=str, default="./results/ebm_training", help="Directory for results and checkpoints.")
    parser.add_argument("--model_name", type=str, default="EBM_scalogram", help="Name for the model and output files.")
    
    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--total_steps", type=int, default=20001, help="Total training steps.")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clipping.")

    # Loss Configuration
    parser.add_argument("--lambda_cd", type=float, default=1.0, help="Coefficient for Contrastive Divergence loss.")
    parser.add_argument("--cd_start_step", type=int, default=5000, help="Step to start applying CD loss.")

    # Checkpoint & Logging
    parser.add_argument("--save_step", type=int, default=2000, help="Checkpoint save and validation frequency.")
    parser.add_argument("--log_step", type=int, default=100, help="Logging frequency.")

    return parser.parse_args()

# --- Validation Function ---
@torch.no_grad()
def validate(model, val_loader, flow_matcher, device):
    model.eval()
    total_val_loss = 0.0
    for i, (x1_val,) in enumerate(val_loader):
        x1_val = x1_val.to(device)
        x0_val = torch.randn_like(x1_val)
        
        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0_val, x1_val)
        vt = model(t, xt)
        
        flow_mse = (vt - ut).square().mean(dim=(1, 2))
        loss_flow = (flow_weight(t) * flow_mse).mean()
        total_val_loss += loss_flow.item()

    avg_val_loss = total_val_loss / len(val_loader)
    model.train()
    return avg_val_loss

# --- Training Loop ---
def train(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {device}")

    # Data
    print(f"Loading data...")
    train_chunks = torch.load(args.train_data_path)
    val_chunks = torch.load(args.val_data_path)
    
    B, C, H, W = train_chunks.shape
    train_chunks_reshaped = train_chunks.permute(0, 2, 3, 1).reshape(train_chunks.shape[0], H * W, C)
    val_chunks_reshaped = val_chunks.permute(0, 2, 3, 1).reshape(val_chunks.shape[0], H * W, C)
    print(f"Data reshaped to (B, L, C): {train_chunks_reshaped.shape}")

    train_dataset = TensorDataset(train_chunks_reshaped, torch.zeros(train_chunks_reshaped.size(0)))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_datalooper = infiniteloop(train_loader)

    val_dataset = TensorDataset(val_chunks_reshaped, torch.zeros(val_chunks_reshaped.size(0)))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = EBM(dim=H * W, channels=C, dim_mults=(1, 2, 4)).to(device)
    ema_model = copy.deepcopy(model).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Optimizer and Flow Matcher
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    flow_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)

    # Training
    print("Starting training...")
    start_time = time.time()
    for step in range(args.total_steps):
        optimizer.zero_grad()

        x1 = next(train_datalooper).to(device)
        x0 = torch.randn_like(x1)

        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
        vt = model(t, xt)
        flow_mse = (vt - ut).square().mean(dim=(1, 2))
        loss_flow = (flow_weight(t) * flow_mse).mean()

        loss_cd = torch.tensor(0.0, device=device)
        if step > args.cd_start_step and args.lambda_cd > 0.0:
            pos_energy = model.potential(x1, torch.ones_like(t)).mean()
            x_neg_init = torch.randn_like(x1)
            at_data_mask = torch.zeros(x1.size(0), dtype=torch.bool, device=device)
            x_neg = gibbs_sampling_time_sweep(x_neg_init, model, at_data_mask)
            neg_energy = model.potential(x_neg, torch.ones_like(t)).mean()
            loss_cd = args.lambda_cd * (pos_energy - neg_energy)

        total_loss = loss_flow + loss_cd
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        ema(model, ema_model, args.ema_decay)

        if step % args.log_step == 0:
            elapsed = time.time() - start_time
            print(f"[Step {step}/{args.total_steps}] Loss: {total_loss.item():.4f} (Flow: {loss_flow.item():.4f}, CD: {loss_cd.item():.4f}) | Time: {elapsed:.2f}s")

        if step > 0 and step % args.save_step == 0:
            val_loss = validate(ema_model, val_loader, flow_matcher, device)
            print(f"--- Validation at Step {step}: Avg Flow Loss = {val_loss:.4f} ---")
            ckpt_path = os.path.join(args.output_dir, f"{args.model_name}_step_{step}.pt")
            torch.save({
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'args': args
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    args = get_args()
    train(args)
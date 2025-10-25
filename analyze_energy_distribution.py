import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os

from ebm_model_vit import EBViTModelWrapper as EBM

def analyze(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading checkpoint from {args.ckpt_path}...")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.ckpt_path}")

    # Use validation chunks to determine model input shape, as it's generally smaller
    val_data_path = "preprocessed_dataset/val_chunks.pt"
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Validation data not found at {val_data_path}. Please ensure preprocessed data exists.")
    
    # We need to load the checkpoint first to get the model args
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt['args']
    
    # Now we can determine shape and initialize the model
    val_chunks_for_shape = torch.load(val_data_path, weights_only=False)
    _B, C, H, W = val_chunks_for_shape.shape
    del val_chunks_for_shape # free memory

    model = EBM(
        dim=(C, H, W),
        num_channels=train_args.num_channels,
        num_res_blocks=train_args.num_res_blocks,
        channel_mult=train_args.channel_mult,
        attention_resolutions=train_args.attention_resolutions,
        num_heads=train_args.num_heads,
        num_head_channels=train_args.num_head_channels,
        dropout=train_args.dropout,
        patch_size=train_args.patch_size,
        embed_dim=train_args.embed_dim,
        transformer_nheads=train_args.transformer_nheads,
        transformer_nlayers=train_args.transformer_nlayers,
        output_scale=train_args.output_scale,
        energy_clamp=train_args.energy_clamp,
    ).to(device)
    model.load_state_dict(ckpt['ema_model'])
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Calculate Energy Scores for Train and Val sets ---
    data_paths = {
        "train": "preprocessed_dataset/train_chunks.pt",
        "val": "preprocessed_dataset/val_chunks.pt"
    }
    energy_scores = {}

    for split, path in data_paths.items():
        print(f"Calculating energy scores for {split} data from {path}...")
        if not os.path.exists(path):
            print(f"Data file not found at {path}, skipping.")
            continue
        
        chunks = torch.load(path, weights_only=False)
        with torch.no_grad():
            # Process in batches to avoid OOM errors on large datasets
            scores_list = []
            batch_size = 256 # A reasonable batch size for inference
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size].to(device).float()
                batch_scores = model.potential(batch_chunks, t=torch.ones(batch_chunks.size(0), device=device))
                scores_list.append(batch_scores.cpu().numpy())
        
        energy_scores[split] = np.concatenate(scores_list)
        print(f"  Mean energy for {split}: {energy_scores[split].mean():.4f}")
        print(f"  Std dev for {split}:  {energy_scores[split].std():.4f}")

    # --- 3. Plot Histograms ---
    if not energy_scores:
        print("No data processed, cannot create plot.")
        return

    print("\nGenerating energy distribution plot...")
    plot_dir = Path("results/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    
    if 'train' in energy_scores:
        plt.hist(energy_scores['train'], bins=100, alpha=0.7, label=f"Train Set (Mean: {energy_scores['train'].mean():.2f})", density=True)
    if 'val' in energy_scores:
        plt.hist(energy_scores['val'], bins=100, alpha=0.7, label=f"Validation Set (Mean: {energy_scores['val'].mean():.2f})", density=True)
        
    plt.title('Energy Score Distribution (Train vs. Validation)')
    plt.xlabel('Energy Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plot_path = plot_dir / "energy_distribution_analysis.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot the energy distribution of a trained EBM model on train and validation sets.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint file.")
    args = parser.parse_args()
    analyze(args)

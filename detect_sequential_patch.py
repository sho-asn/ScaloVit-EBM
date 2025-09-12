import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import os
from glob import glob
import numpy as np
from tqdm import tqdm

# --- Model & Data Handling Classes (Copied from train_sequential_patch.py) ---
from train_sequential_patch import SlidingWindowPatchDataset, PatchSequenceTransformer
from metrics import compute_all_metrics
from utils.utils_visualization import plot_energy_with_anomalies

# --- Argument Parsing ---
def get_args():
    parser = argparse.ArgumentParser(description="Evaluate the patch-aware sequential model.")
    parser.add_argument("--stage_b_ckpt_path", type=str, required=True, help="Path to the trained Stage B (Patch-Aware Sequence) model checkpoint.")
    parser.add_argument("--features_dir", type=str, default="./results/features", help="Directory containing the extracted feature files (*.pt).")
    parser.add_argument("--output_dir", type=str, default="./results/final_patch_detection", help="Directory to save plots and results.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing data.")
    parser.add_argument("--threshold_percentile", type=float, default=99.0, help="Percentile of validation scores to use as anomaly threshold.")
    return parser.parse_args()

# --- Main Detection Logic ---

def get_anomaly_scores(model, feature_files, sequence_length, batch_size, device, desc):
    """Helper function to compute anomaly scores (MSE) for a feature set."""
    dataset = SlidingWindowPatchDataset(feature_files, sequence_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_scores_mean = []
    all_scores_per_patch = []
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for x_seq, y_true in tqdm(loader, desc=desc):
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            y_pred = model(x_seq)
            
            # Calculate loss for the whole chunk representation (global + patches)
            # Shape: (B, 1+N, E)
            loss_tensor = criterion(y_pred, y_true)
            
            # 1. Per-patch scores: (B, N)
            # We only care about the patches, not the global token, for anomaly localization
            patch_loss = loss_tensor[:, 1:, :].mean(dim=2) # Average MSE over the embedding dimension
            all_scores_per_patch.append(patch_loss.cpu())
            
            # 2. Mean score for thresholding and 1D plotting: (B,)
            # Average the loss over all tokens (global + patches) and the embedding dim
            mean_chunk_loss = loss_tensor.mean(dim=[1, 2])
            all_scores_mean.append(mean_chunk_loss.cpu())
            
    return torch.cat(all_scores_mean).numpy(), torch.cat(all_scores_per_patch).numpy()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    # --- Load Stage B (Sequence) Model and its args ---
    print(f"Loading Stage B checkpoint from {args.stage_b_ckpt_path}...")
    ckpt_b = torch.load(args.stage_b_ckpt_path, map_location=device, weights_only=False)
    model_args_b = ckpt_b['args']
    num_patches = ckpt_b['num_patches']

    # Infer embed_dim directly from the model's state dict for robustness
    embed_dim = ckpt_b['model_state_dict']['input_proj.weight'].shape[1]
    print(f"Inferred from model checkpoint -> Num Patches: {num_patches}, Embedding Dim: {embed_dim}")

    model_b = PatchSequenceTransformer(
        embed_dim=embed_dim,
        num_patches=num_patches,
        num_layers=model_args_b.num_layers,
        num_heads=model_args_b.num_heads,
        hidden_dim_multiplier=model_args_b.hidden_dim_multiplier,
        dropout=model_args_b.dropout
    ).to(device)
    model_b.load_state_dict(ckpt_b['model_state_dict'])
    model_b.eval()
    print("Stage B patch-aware model loaded successfully.")

    # --- Establish Anomaly Threshold from Validation Data (PATCH-BASED) ---
    print("Processing validation set to establish anomaly threshold...")
    val_feature_files = sorted(glob(os.path.join(args.features_dir, "*val*.pt")))
    if not val_feature_files:
        raise FileNotFoundError(f"No validation feature files found in {args.features_dir}")

    _, val_scores_per_patch = get_anomaly_scores(
        model_b, val_feature_files, model_args_b.sequence_length, args.batch_size, device, "Stage B: Scoring Validation Set"
    )
    # Create a single threshold based on the distribution of all patch scores
    patch_anomaly_threshold = np.percentile(val_scores_per_patch.flatten(), args.threshold_percentile)
    print(f"Patch-level anomaly threshold set to {patch_anomaly_threshold:.6f} ({args.threshold_percentile}th percentile of all patch scores)")

    # --- Evaluate on Test Sets (PATCH-BASED) ---
    test_feature_files = sorted([f for f in glob(os.path.join(args.features_dir, "*test*.pt"))])

    for feature_path in test_feature_files:
        set_name = Path(feature_path).stem
        print(f"\n--- Evaluating {set_name} ---")
        
        # Load original metadata from the feature file
        original_data = torch.load(feature_path, weights_only=False)
        ground_truth_per_patch = original_data['labels'] # This is 1D, per-patch-window
        num_chunks_in_file = original_data['patch_embeddings'].shape[0]

        # Get anomaly scores for the test set
        _, test_scores_per_patch = get_anomaly_scores(
            model_b, [feature_path], model_args_b.sequence_length, args.batch_size, device, "Stage B: Scoring Test Set"
        )

        # --- Save Raw Per-Patch Scores ---
        per_patch_scores_path = output_dir / f"{set_name}_per_patch_scores.npy"
        np.save(per_patch_scores_path, test_scores_per_patch)
        print(f"Saved per-patch anomaly scores to {per_patch_scores_path}")

        # --- Align Scores and GT at the PATCH level ---
        predicted_patch_scores = test_scores_per_patch.flatten()

        # Infer how many patch-windows from the GT correspond to one chunk.
        patches_per_chunk_in_time = 0
        if num_chunks_in_file > 0:
            patches_per_chunk_in_time = round(len(ground_truth_per_patch) / num_chunks_in_file)
        
        # The model's scores start after sequence_length chunks have passed.
        # We must offset the ground truth by the same amount in patches.
        start_chunk_idx = model_args_b.sequence_length
        start_patch_idx = start_chunk_idx * patches_per_chunk_in_time

        num_predicted_patches = len(predicted_patch_scores)
        aligned_gt_per_patch = ground_truth_per_patch[start_patch_idx : start_patch_idx + num_predicted_patches]

        # Create binary predictions based on the patch-level threshold
        predicted_anomalies_per_patch = (predicted_patch_scores > patch_anomaly_threshold).astype(int)

        # --- Calculate and Display Metrics ---
        if len(aligned_gt_per_patch) != len(predicted_anomalies_per_patch):
            print(f"Warning: Length mismatch between ground truth ({len(aligned_gt_per_patch)}) and predictions ({len(predicted_anomalies_per_patch)}). Truncating for metrics.")
            min_len = min(len(aligned_gt_per_patch), len(predicted_anomalies_per_patch))
            aligned_gt_per_patch = aligned_gt_per_patch[:min_len]
            predicted_anomalies_per_patch = predicted_anomalies_per_patch[:min_len]
            predicted_patch_scores = predicted_patch_scores[:min_len]

        metrics = compute_all_metrics(aligned_gt_per_patch.numpy(), predicted_anomalies_per_patch, predicted_patch_scores)
        print("Computed Metrics (based on per-patch scores):")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")

        # --- Plot Per-Patch Scores ---
        plot_path = output_dir / f"{set_name}_per_patch_scores_plot.png"
        plot_energy_with_anomalies(
            energy_scores=predicted_patch_scores,
            threshold=patch_anomaly_threshold,
            save_path=plot_path,
            title=f"Per-Patch Anomaly Scores for {set_name}",
            ground_truth_labels=aligned_gt_per_patch.numpy(),
        )
        print(f"Per-patch scores plot saved to {plot_path}")

if __name__ == "__main__":
    main()

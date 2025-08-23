import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
from glob import glob

from sklearn.metrics import roc_auc_score
from ebm_model_vit import EBViTModelWrapper as EBM

# --- Main Detection Logic ---
def detect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading checkpoint from {args.ckpt_path}...")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.ckpt_path}")
        
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt['args']
    
    # Load validation chunks to determine model input shape
    val_chunks = torch.load(args.val_data_path)
    _B, C, H, W = val_chunks.shape

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

    # --- 2. Establish Anomaly Threshold from Validation Data ---
    print(f"Calculating energy scores on validation data from {args.val_data_path}...")
    with torch.no_grad():
        val_scores = model.potential(val_chunks.to(device), t=torch.ones(val_chunks.size(0), device=device))
        val_scores = val_scores.cpu().numpy()
    
    anomaly_threshold = np.percentile(val_scores, args.threshold_percentile)
    print(f"Anomaly threshold set to {anomaly_threshold:.4f}")

    # --- 3. Evaluate on Preprocessed Test Sets ---
    test_files = sorted(glob(os.path.join(args.test_data_dir, "test_*.pt")))
    if not test_files:
        print(f"No preprocessed test files found in {args.test_data_dir}. Please run preprocess_data.py first.")
        return

    for test_file_path in test_files:
        set_name = Path(test_file_path).stem
        print(f"\n--- Evaluating {set_name} ---")
        
        data = torch.load(test_file_path, weights_only=False)
        test_chunks = data['chunks']
        ground_truth = data['labels']

        if test_chunks.shape[0] == 0:
            print("No chunks in this file, skipping.")
            continue

        with torch.no_grad():
            test_scores = model.potential(test_chunks.to(device), t=torch.ones(test_chunks.size(0), device=device))
            test_scores = test_scores.cpu().numpy()

        if len(np.unique(ground_truth)) > 1:
            auc_score = roc_auc_score(ground_truth, test_scores)
            print(f"ROC AUC Score: {auc_score:.4f}")
        else:
            print("Ground truth contains only one class, cannot calculate AUC.")

        # --- Calculate Detection and False Alarm Rates ---
        predicted_anomalies = test_scores > anomaly_threshold
        
        faulty_period_indices = np.where(ground_truth == 1)[0]
        non_faulty_period_indices = np.where(ground_truth == 0)[0]

        if len(faulty_period_indices) > 0:
            correctly_identified_faults = np.sum(predicted_anomalies[faulty_period_indices])
            detection_rate = correctly_identified_faults / len(faulty_period_indices)
            print(f"Detection Rate (Recall): {detection_rate:.4f}")
        else:
            print("No faulty period in ground truth, cannot calculate Detection Rate.")

        if len(non_faulty_period_indices) > 0:
            incorrectly_flagged_non_faults = np.sum(predicted_anomalies[non_faulty_period_indices])
            false_alarm_rate = incorrectly_flagged_non_faults / len(non_faulty_period_indices)
            print(f"False Alarm Rate: {false_alarm_rate:.4f}")
        else:
            print("No non-faulty period in ground truth, cannot calculate False Alarm Rate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an EBM model for anomaly detection on preprocessed data.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--val_data_path", type=str, default="preprocessed_dataset/val_chunks.pt", help="Path to the validation data for setting the anomaly threshold.")
    parser.add_argument("--test_data_dir", type=str, default="preprocessed_dataset", help="Directory containing the preprocessed test set files (*.pt).")
    parser.add_argument("--threshold_percentile", type=float, default=95, help="Percentile of validation scores to use as anomaly threshold.")
    args = parser.parse_args()
    detect(args)

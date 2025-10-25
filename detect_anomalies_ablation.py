# File: detect_anomalies_ablation.py
# An evaluation script for running ablation studies.

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
from glob import glob
from scipy.ndimage import uniform_filter1d

import json
import pandas as pd

# Import all model types
from scalovit.models import (
    EBViTModelWrapper as PatchBasedEBM,
    ImageBased_EBViTModelWrapper,
    ConvHead_EBMWrapper,
)

from scalovit.metrics import compute_all_metrics, compute_metrics_from_cm, calculate_roc_auc
from scalovit.scoring import detect_with_ema, detect_with_cusum
from scalovit.transforms import WAVEmbedder
from scalovit.utils.visualization import plot_energy_with_anomalies

# --- New Reconstruction function for Image-Based scores ---
def reconstruct_from_chunk_scores(scores, signal_len, chunk_width, stride, patch_width, agg_method='max'):
    num_signal_patches = signal_len // patch_width
    if agg_method == 'max':
        score_agg = np.full(num_signal_patches, -np.inf, dtype=np.float32)
    else:
        score_sum = np.zeros(num_signal_patches, dtype=np.float32)
        score_count = np.zeros(num_signal_patches, dtype=np.int32)

    covered_patches = np.zeros(num_signal_patches, dtype=bool)
    num_chunks = scores.shape[0]

    for i in range(num_chunks):
        chunk_score = scores[i]
        chunk_start_time = i * stride
        start_patch_idx = chunk_start_time // patch_width
        end_patch_idx = (chunk_start_time + chunk_width) // patch_width
        for j in range(start_patch_idx, end_patch_idx):
            if j < num_signal_patches:
                if agg_method == 'max':
                    score_agg[j] = np.maximum(score_agg[j], chunk_score)
                else:
                    score_sum[j] += chunk_score
                    score_count[j] += 1
                covered_patches[j] = True

    if agg_method == 'max':
        final_scores = np.where(covered_patches, score_agg, 0)
    else:
        final_scores = np.divide(score_sum, score_count, out=np.zeros_like(score_sum), where=score_count != 0)

    if not np.any(covered_patches):
        return np.array([])
    last_covered_idx = np.where(covered_patches)[0][-1]
    return final_scores[:last_covered_idx + 1]

# Original reconstruction function
def reconstruct_scores_from_overlapping_chunks(scores, signal_len, chunk_width, stride, patch_size, image_height, agg_method='max'):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    patch_height, patch_width = patch_size
    patches_per_chunk_h = image_height // patch_height
    patches_per_chunk_w = chunk_width // patch_width
    num_signal_patches = signal_len // patch_width

    if agg_method == 'max':
        score_agg = np.full(num_signal_patches, -np.inf, dtype=np.float32)
    else:
        score_sum = np.zeros(num_signal_patches, dtype=np.float32)
        score_count = np.zeros(num_signal_patches, dtype=np.int32)

    covered_patches = np.zeros(num_signal_patches, dtype=bool)
    num_chunks = scores.shape[0]

    for i in range(num_chunks):
        chunk_scores_grid = scores[i].reshape((patches_per_chunk_h, patches_per_chunk_w))
        time_patch_scores = chunk_scores_grid.mean(axis=0)
        chunk_start_time = i * stride
        for j in range(patches_per_chunk_w):
            global_patch_idx = (chunk_start_time // patch_width) + j
            if global_patch_idx < num_signal_patches:
                if agg_method == 'max':
                    score_agg[global_patch_idx] = np.maximum(score_agg[global_patch_idx], time_patch_scores[j])
                else:
                    score_sum[global_patch_idx] += time_patch_scores[j]
                    score_count[global_patch_idx] += 1
                covered_patches[global_patch_idx] = True

    if agg_method == 'max':
        final_scores = np.where(covered_patches, score_agg, 0)
    else:
        final_scores = np.divide(score_sum, score_count, out=np.zeros_like(score_sum), where=score_count != 0)

    if not np.any(covered_patches):
        return np.array([])
    last_covered_idx = np.where(covered_patches)[0][-1]
    return final_scores[:last_covered_idx + 1]

def detect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading checkpoint from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt['args']
    val_chunks_for_shape = torch.load(args.val_data_path)
    if isinstance(val_chunks_for_shape, dict):
        val_chunks_for_shape = val_chunks_for_shape['chunks']
    _B, C, H, W = val_chunks_for_shape.shape

    # Select model based on ablation type
    model_class = PatchBasedEBM
    if args.ablation_model_type == 'image_based':
        print("--- Loading ABLATION model: ImageBased_EBViTModelWrapper ---")
        model_class = ImageBased_EBViTModelWrapper
    elif args.ablation_model_type == 'conv_head':
        print("--- Loading ABLATION model: ConvHead_EBMWrapper ---")
        model_class = ConvHead_EBMWrapper
    else:
        print("--- Loading MAIN model: PatchBasedEBM ---")

    model_args_dict = vars(train_args)
    model_args_dict['dim'] = (C, H, W)

    # Define keys that are for the training script, not the model architecture
    non_model_keys = [
        # Script/run config
        'ablation_model_type', 'train_data_path', 'val_data_path', 'output_dir',
        'model_name', 'resume_ckpt', 'save_step', 'log_step',
        # Training loop config
        'lr', 'batch_size', 'total_steps', 'warmup', 'ema_decay', 'grad_clip',
        'num_workers', 'use_amp', 'gradient_accumulation_steps', 'no_shuffle',
        # Loss function config
        'lambda_cd', 'time_cutoff', 'cd_neg_clamp', 'cd_trim_fraction', 'lambda_gp', 'lambda_smooth',
        # Gibbs/CD sampling config
        'n_gibbs', 'dt_gibbs', 'epsilon_max', 'split_negative', 'same_temperature_scheduler'
    ]
    
    # Create a clean dictionary for the model constructor
    clean_model_args = {k: v for k, v in model_args_dict.items() if k not in non_model_keys}

    model = model_class(**clean_model_args).to(device)
    model.load_state_dict(ckpt['ema_model'])
    model.eval()
    print("Model loaded successfully.")

    # --- The rest of the script is largely the same ---
    # --- 2. Setup for Scoring ---
    is_mag_only = "mag" in args.val_data_path
    embedder = None
    if args.use_residual_scoring:
        print("Residual scoring enabled...")
        suffix = "_wavelet_mag" if is_mag_only else "_wavelet"
        norm_params_path = Path(args.test_data_dir) / f"norm_params{suffix}.pt"
        norm_params = torch.load(norm_params_path)
        embedder = WAVEmbedder(
            device=device, seq_len=W,
            wavelet_name=args.wavelet_name, 
            scales_arange=(args.wavelet_scales_min, args.wavelet_scales_max)
        )
        embedder.min_mag = norm_params['min_mag'].to(device)
        embedder.max_mag = norm_params['max_mag'].to(device)
        if not is_mag_only:
            embedder.min_phase = norm_params['min_phase'].to(device)
            embedder.max_phase = norm_params['max_phase'].to(device)

    # --- 3. Establish Anomaly Threshold from Validation Data ---
    print(f"Calculating scores on validation data from {args.val_data_path}...")
    val_data = torch.load(args.val_data_path)
    val_scores = []
    with torch.no_grad():
        val_chunks_to_process = val_data['raw_chunks'] if isinstance(val_data, dict) and args.use_residual_scoring else val_data
        for i in range(0, len(val_chunks_to_process), args.batch_size):
            batch = val_chunks_to_process[i:i+args.batch_size]
            if args.use_residual_scoring:
                batch_np = batch.numpy()
                residual_batch = np.empty_like(batch_np)
                for j in range(batch_np.shape[0]):
                    trend = uniform_filter1d(batch_np[j], size=args.detrend_window_size, axis=0, mode='nearest')
                    residual_batch[j] = batch_np[j] - trend
                scalogram_batch = embedder.ts_to_img(torch.from_numpy(residual_batch).float().to(device))
                if is_mag_only:
                    scalogram_batch = scalogram_batch[:, ::2, :, :]
                scores_input = scalogram_batch
            else:
                scores_input = batch.to(device).float()
            # The model's potential function is called here
            scores_per_chunk_or_patch = model.potential(scores_input, t=torch.ones(scores_input.size(0), device=device))
            # For patch-based, we take the mean to get one score per chunk for thresholding
            if scores_per_chunk_or_patch.ndim > 1:
                val_scores.append(scores_per_chunk_or_patch.mean(dim=1).cpu())
            else:
                val_scores.append(scores_per_chunk_or_patch.cpu())
        val_scores = torch.cat(val_scores, dim=0).numpy()
    
    # --- The rest of the script is identical to your original, but I will copy it here for completeness ---
    print(f"Tuning threshold for '{args.scoring_method}' method...")
    if args.scoring_method == 'threshold':
        anomaly_threshold = np.percentile(val_scores, args.threshold_percentile)
        print(f"Static energy threshold set to {anomaly_threshold:.4f}")
    elif args.scoring_method == 'ema':
        _, ema_values = detect_with_ema(val_scores, alpha=args.ema_alpha)
        anomaly_threshold = np.percentile(ema_values, args.threshold_percentile)
        print(f"EMA threshold set to {anomaly_threshold:.4f}")
    elif args.scoring_method == 'cusum':
        cusum_baseline = np.median(val_scores)
        std_dev_val = np.std(val_scores)
        cusum_k_value = args.cusum_k * std_dev_val
        _, val_cusum_values = detect_with_cusum(val_scores, baseline=cusum_baseline, h=float('inf'), k=cusum_k_value)
        anomaly_threshold = np.percentile(val_cusum_values, args.threshold_percentile)
        print(f"CUSUM baseline set to {cusum_baseline:.4f}, k set to {cusum_k_value:.4f}, threshold (h) set to {anomaly_threshold:.4f}")

    # --- 4. Evaluate on Preprocessed Test Sets ---
    is_residual = "_residual" in args.val_data_path
    test_files = sorted(glob(os.path.join(args.test_data_dir, "test_*.pt")))
    test_files = [f for f in test_files if ('_mag' in Path(f).name) == is_mag_only and \
                                           ('_residual' in Path(f).name) == is_residual and \
                                           '_stft' not in Path(f).name]

    # Create threshold info dict for reporting
    threshold_info = {
        "value": float(anomaly_threshold),
        "percentile": args.threshold_percentile,
        "method": args.scoring_method
    }

    # Initialize results dict with threshold info
    all_results = {"threshold": threshold_info}
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    all_ground_truth = []
    all_final_scores = []

    for test_file_path in test_files:
        set_name = Path(test_file_path).stem
        print(f"\n--- Evaluating {set_name} ---")
        data = torch.load(test_file_path, weights_only=False)
        ground_truth, signal_len, stride = data['labels'], data['signal_len'], data['stride']
        chunks_to_process = data['raw_chunks'] if args.use_residual_scoring else data['chunks']
        if chunks_to_process.shape[0] == 0:
            print("No chunks in this file, skipping."); continue

        all_test_scores = []
        with torch.no_grad():
            for i in range(0, len(chunks_to_process), args.batch_size):
                batch = chunks_to_process[i:i+args.batch_size]
                if args.use_residual_scoring:
                    batch_np = batch.numpy()
                    residual_batch = np.empty_like(batch_np)
                    for j in range(batch_np.shape[0]):
                        trend = uniform_filter1d(batch_np[j], size=args.detrend_window_size, axis=0, mode='nearest')
                        residual_batch[j] = batch_np[j] - trend
                    scalogram_batch = embedder.ts_to_img(torch.from_numpy(residual_batch).float().to(device))
                    if is_mag_only:
                        scalogram_batch = scalogram_batch[:, ::2, :, :]
                    scores_input = scalogram_batch
                else:
                    scores_input = batch.to(device).float()
                scores = model.potential(scores_input, t=torch.ones(scores_input.size(0), device=device))
                all_test_scores.append(scores)
            test_scores = torch.cat(all_test_scores, dim=0).cpu().numpy()

        if test_scores.ndim == 1:
            final_scores = reconstruct_from_chunk_scores(test_scores, signal_len, W, stride, train_args.patch_size[1], args.overlap_agg_method)
        else:
            final_scores = reconstruct_scores_from_overlapping_chunks(test_scores, signal_len, W, stride, train_args.patch_size, H, args.overlap_agg_method)

        common_len = min(len(final_scores), len(ground_truth))
        final_scores, ground_truth_np = final_scores[:common_len], ground_truth[:common_len].numpy()

        if args.scoring_method == 'threshold':
            predicted_anomalies = (final_scores > anomaly_threshold).astype(int)
            scores_to_plot, scores_for_metrics = final_scores, final_scores
        elif args.scoring_method == 'ema':
            alarms, ema_values = detect_with_ema(final_scores, alpha=args.ema_alpha, threshold=anomaly_threshold)
            predicted_anomalies, scores_to_plot, scores_for_metrics = np.array(alarms).astype(int), ema_values, ema_values
        elif args.scoring_method == 'cusum':
            std_dev_val = np.std(val_scores)
            cusum_k_value = args.cusum_k * std_dev_val
            alarms, cusum_values = detect_with_cusum(final_scores, baseline=cusum_baseline, h=anomaly_threshold, k=cusum_k_value)
            predicted_anomalies, scores_to_plot, scores_for_metrics = np.array(alarms).astype(int), cusum_values, cusum_values

        plot_dir = Path(args.output_dir) / set_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"score_plot_{args.scoring_method}.png"
        plot_energy_with_anomalies(scores_to_plot, anomaly_threshold, plot_path, f"{args.scoring_method.upper()} Scores for {set_name}", ground_truth_np)
        print(f"Scoring plot saved to {plot_path}")
        metrics = compute_all_metrics(ground_truth_np, predicted_anomalies, scores_for_metrics)
        all_results[set_name] = metrics

        # Accumulate for overall results
        total_tp += metrics["TP"]
        total_tn += metrics["TN"]
        total_fp += metrics["FP"]
        total_fn += metrics["FN"]
        all_ground_truth.append(ground_truth_np)
        all_final_scores.append(scores_for_metrics)

        # --- Save scores to CSV if requested ---
        if args.save_scores:
            df_data = {
                'index': np.arange(len(ground_truth_np)),
                'ground_truth': ground_truth_np,
                'score': scores_for_metrics,
                'prediction': predicted_anomalies
            }
            scores_df = pd.DataFrame(df_data)
            csv_save_path = plot_dir / f"scores_{args.scoring_method}.csv"
            scores_df.to_csv(csv_save_path, index=False)
            print(f"Scores saved to {csv_save_path}")

        print("Computed Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"  {metric_name}: {metric_value}")

    # --- 5. Compute Overall Metrics and Save Results ---
    if all_results:
        print("\n--- Computing Overall Metrics ---")
        overall_metrics = compute_metrics_from_cm(total_tp, total_tn, total_fp, total_fn)
        
        # Calculate overall ROC_AUC
        overall_gt = np.concatenate(all_ground_truth)
        overall_scores = np.concatenate(all_final_scores)
        overall_roc_auc = calculate_roc_auc(overall_gt, overall_scores)
        overall_metrics["ROC_AUC"] = overall_roc_auc
        
        all_results["overall"] = overall_metrics
        
        print("Overall Metrics:")
        for metric_name, metric_value in overall_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"  {metric_name}: {metric_value}")

        # Save to JSON
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_output_path = output_dir / "detection_metrics.json"
        with open(json_output_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nAll results saved to {json_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate EBMs for anomaly detection.")
    # Add the new argument for selecting the model
    parser.add_argument("--ablation_model_type", type=str, default='patch_based', choices=['patch_based', 'image_based', 'conv_head'], help="Type of model architecture to use for ablation.")
    # Keep all other arguments the same
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--val_data_path", type=str, default="preprocessed_dataset/val_chunks_wavelet_mag.pt")
    parser.add_argument("--test_data_dir", type=str, default="preprocessed_dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--scoring_method", type=str, default="threshold", choices=["threshold", "ema", "cusum"])
    parser.add_argument("--threshold_percentile", type=float, default=99.5)
    parser.add_argument("--overlap_agg_method", type=str, default="max", choices=["max", "mean"])
    parser.add_argument("--use_residual_scoring", action="store_true")
    parser.add_argument("--detrend_window_size", type=int, default=256)
    parser.add_argument("--wavelet_name", type=str, default="morl")
    parser.add_argument("--wavelet_scales_min", type=int, default=1)
    parser.add_argument("--wavelet_scales_max", type=int, default=129)
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--cusum_k", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="results/ablation_run", help="Directory to save all outputs (plots, scores, and metrics).")
    parser.add_argument("--save_scores", action="store_true", help="If set, save the computed scores and labels to a CSV file for each test set.")
    args = parser.parse_args()
    detect(args)

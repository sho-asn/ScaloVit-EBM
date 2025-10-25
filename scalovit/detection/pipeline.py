"""Inference-time utilities for running anomaly detection with trained models."""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter1d

from scalovit.metrics import calculate_roc_auc, compute_all_metrics, compute_metrics_from_cm
from scalovit.models import EBViTModelWrapper as EBM
from scalovit.scoring import detect_with_cusum, detect_with_ema
from scalovit.transforms import WAVEmbedder
from scalovit.utils.visualization import plot_energy_with_anomalies


def reconstruct_scores_from_overlapping_chunks(
    scores: np.ndarray,
    signal_len: int,
    chunk_width: int,
    stride: int,
    patch_size: tuple[int, int] | int,
    image_height: int,
    agg_method: str = "max",
) -> np.ndarray:
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    patch_height, patch_width = patch_size

    patches_per_chunk_h = image_height // patch_height
    patches_per_chunk_w = chunk_width // patch_width
    num_signal_patches = signal_len // patch_width

    if agg_method == "max":
        score_agg = np.full(num_signal_patches, -np.inf, dtype=np.float32)
    else:
        score_sum = np.zeros(num_signal_patches, dtype=np.float32)
        score_count = np.zeros(num_signal_patches, dtype=np.int32)

    covered_patches = np.zeros(num_signal_patches, dtype=bool)
    num_chunks = scores.shape[0]

    for chunk_idx in range(num_chunks):
        chunk_scores_grid = scores[chunk_idx].reshape((patches_per_chunk_h, patches_per_chunk_w))
        time_patch_scores = chunk_scores_grid.mean(axis=0)
        chunk_start_time = chunk_idx * stride

        for patch_idx in range(patches_per_chunk_w):
            global_patch_idx = (chunk_start_time // patch_width) + patch_idx
            if global_patch_idx < num_signal_patches:
                if agg_method == "max":
                    score_agg[global_patch_idx] = np.maximum(score_agg[global_patch_idx], time_patch_scores[patch_idx])
                else:
                    score_sum[global_patch_idx] += time_patch_scores[patch_idx]
                    score_count[global_patch_idx] += 1
                covered_patches[global_patch_idx] = True

    if agg_method == "max":
        final_scores = np.where(covered_patches, score_agg, 0)
    else:
        final_scores = np.divide(score_sum, score_count, out=np.zeros_like(score_sum), where=score_count != 0)

    if np.any(covered_patches):
        last_covered_idx = np.where(covered_patches)[0][-1]
    else:
        return np.array([])

    return final_scores[: last_covered_idx + 1]


def detect(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {args.ckpt_path}...")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.ckpt_path}")

    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt["args"]

    val_chunks_for_shape = torch.load(args.val_data_path)
    if isinstance(val_chunks_for_shape, dict):
        val_chunks_for_shape = val_chunks_for_shape["chunks"]
    _, channels, height, width = val_chunks_for_shape.shape

    model = EBM(
        dim=(channels, height, width),
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
    model.load_state_dict(ckpt["ema_model"])
    model.eval()
    print("Model loaded successfully.")

    is_mag_only = "mag" in args.val_data_path
    embedder = None
    if args.use_residual_scoring:
        print("Residual scoring enabled. Loading CWT embedder and normalization params...")
        suffix = "_wavelet_mag" if is_mag_only else "_wavelet"
        norm_params_path = Path(args.test_data_dir) / f"norm_params{suffix}.pt"
        if not norm_params_path.exists():
            raise FileNotFoundError(f"Normalization parameters not found at {norm_params_path}. Please run preprocess_data.py.")
        norm_params = torch.load(norm_params_path)

        embedder = WAVEmbedder(
            device=device,
            seq_len=width,
            wavelet_name=args.wavelet_name,
            scales_arange=(args.wavelet_scales_min, args.wavelet_scales_max),
        )
        embedder.min_mag = norm_params["min_mag"].to(device)
        embedder.max_mag = norm_params["max_mag"].to(device)
        if not is_mag_only:
            embedder.min_phase = norm_params["min_phase"].to(device)
            embedder.max_phase = norm_params["max_phase"].to(device)

    print(f"Calculating scores on validation data from {args.val_data_path}...")
    val_data = torch.load(args.val_data_path)
    val_scores: List[torch.Tensor] = []
    with torch.no_grad():
        val_chunks_to_process = val_data["raw_chunks"] if isinstance(val_data, dict) and args.use_residual_scoring else val_data
        for start_idx in range(0, len(val_chunks_to_process), args.batch_size):
            batch = val_chunks_to_process[start_idx : start_idx + args.batch_size]

            if args.use_residual_scoring:
                batch_np = batch.numpy()
                residual_batch = np.empty_like(batch_np)
                for idx in range(batch_np.shape[0]):
                    trend = uniform_filter1d(batch_np[idx], size=args.detrend_window_size, axis=0, mode="nearest")
                    residual_batch[idx] = batch_np[idx] - trend

                scalogram_batch = embedder.ts_to_img(torch.from_numpy(residual_batch).float().to(device))
                if is_mag_only:
                    scalogram_batch = scalogram_batch[:, ::2, :, :]
                scores_input = scalogram_batch
            else:
                scores_input = batch.to(device).float()

            scores_per_patch = model.potential(scores_input, t=torch.ones(scores_input.size(0), device=device))
            val_scores.append(scores_per_patch.mean(dim=1).cpu())

        val_scores = torch.cat(val_scores, dim=0).numpy()

    print(f"Tuning threshold for '{args.scoring_method}' method...")
    cusum_baseline = None
    cusum_k_value = None
    if args.scoring_method == "threshold":
        anomaly_threshold = np.percentile(val_scores, args.threshold_percentile)
        print(f"Static energy threshold set to {anomaly_threshold:.4f}")
    elif args.scoring_method == "ema":
        _, ema_values = detect_with_ema(val_scores, alpha=args.ema_alpha)
        anomaly_threshold = np.percentile(ema_values, args.threshold_percentile)
        print(f"EMA threshold set to {anomaly_threshold:.4f}")
    elif args.scoring_method == "cusum":
        cusum_baseline = np.median(val_scores)
        std_dev_val = np.std(val_scores)
        cusum_k_value = args.cusum_k * std_dev_val
        _, val_cusum_values = detect_with_cusum(val_scores, baseline=cusum_baseline, h=float("inf"), k=cusum_k_value)
        anomaly_threshold = np.percentile(val_cusum_values, args.threshold_percentile)
        print(f"CUSUM baseline set to {cusum_baseline:.4f}, k set to {cusum_k_value:.4f}, threshold (h) set to {anomaly_threshold:.4f}")
    else:
        raise ValueError(f"Unknown scoring method: {args.scoring_method}")

    is_residual = "_residual" in args.val_data_path
    test_files = sorted(glob(os.path.join(args.test_data_dir, "test_*.pt")))
    test_files = [
        path
        for path in test_files
        if ("_mag" in Path(path).name) == is_mag_only
        and ("_residual" in Path(path).name) == is_residual
        and "_stft" not in Path(path).name
    ]

    if not test_files:
        print(f"No preprocessed test files found in {args.test_data_dir} matching the model type.")
        return {}

    threshold_info = {"value": float(anomaly_threshold), "percentile": args.threshold_percentile, "method": args.scoring_method}
    all_results: Dict[str, Dict[str, float]] = {"threshold": threshold_info}
    total_tp = total_tn = total_fp = total_fn = 0
    all_ground_truth: List[np.ndarray] = []
    all_final_scores: List[np.ndarray] = []

    for test_file_path in test_files:
        set_name = Path(test_file_path).stem
        print(f"\n--- Evaluating {set_name} ---")

        data = torch.load(test_file_path, weights_only=False)
        ground_truth = data["labels"]
        signal_len = data["signal_len"]
        stride = data["stride"]
        chunks_to_process = data["raw_chunks"] if args.use_residual_scoring else data["chunks"]
        if chunks_to_process.shape[0] == 0:
            print("No chunks in this file, skipping.")
            continue

        all_test_scores_per_patch: List[torch.Tensor] = []
        with torch.no_grad():
            for start_idx in range(0, len(chunks_to_process), args.batch_size):
                batch = chunks_to_process[start_idx : start_idx + args.batch_size]

                if args.use_residual_scoring:
                    batch_np = batch.numpy()
                    residual_batch = np.empty_like(batch_np)
                    for idx in range(batch_np.shape[0]):
                        trend = uniform_filter1d(batch_np[idx], size=args.detrend_window_size, axis=0, mode="nearest")
                        residual_batch[idx] = batch_np[idx] - trend

                    scalogram_batch = embedder.ts_to_img(torch.from_numpy(residual_batch).float().to(device))
                    if is_mag_only:
                        scalogram_batch = scalogram_batch[:, ::2, :, :]
                    scores_input = scalogram_batch
                else:
                    scores_input = batch.to(device).float()

                scores_per_patch = model.potential(scores_input, t=torch.ones(scores_input.size(0), device=device))
                all_test_scores_per_patch.append(scores_per_patch)

            test_scores_per_patch = torch.cat(all_test_scores_per_patch, dim=0).cpu().numpy()

        final_scores = reconstruct_scores_from_overlapping_chunks(
            scores=test_scores_per_patch,
            signal_len=signal_len,
            chunk_width=width,
            stride=stride,
            patch_size=train_args.patch_size,
            image_height=height,
            agg_method=args.overlap_agg_method,
        )

        common_len = min(len(final_scores), len(ground_truth))
        final_scores = final_scores[:common_len]
        ground_truth_np = ground_truth[:common_len].numpy()

        if args.scoring_method == "threshold":
            predicted_anomalies = (final_scores > anomaly_threshold).astype(int)
            plot_title = f"Energy Scores for {set_name}"
            scores_to_plot = final_scores
            scores_for_metrics = final_scores
        elif args.scoring_method == "ema":
            alarms, ema_values = detect_with_ema(final_scores, alpha=args.ema_alpha, threshold=anomaly_threshold)
            predicted_anomalies = np.array(alarms).astype(int)
            plot_title = f"EMA({args.ema_alpha}) Scores for {set_name}"
            scores_to_plot = ema_values
            scores_for_metrics = ema_values
        else:  # cusum
            alarms, cusum_values = detect_with_cusum(final_scores, baseline=cusum_baseline, h=anomaly_threshold, k=cusum_k_value)
            predicted_anomalies = np.array(alarms).astype(int)
            plot_title = f"CUSUM (h={anomaly_threshold:.2f}) Scores for {set_name}"
            scores_to_plot = cusum_values
            scores_for_metrics = cusum_values

        plot_dir = Path(args.output_dir) / set_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"score_plot_{args.scoring_method}.png"
        plot_energy_with_anomalies(
            energy_scores=scores_to_plot,
            threshold=anomaly_threshold,
            save_path=plot_path,
            title=plot_title,
            ground_truth_labels=ground_truth_np,
        )
        print(f"Scoring plot saved to {plot_path}")

        if args.save_scores:
            df_data = {
                "index": np.arange(len(ground_truth_np)),
                "ground_truth": ground_truth_np,
                "score": scores_for_metrics,
                "prediction": predicted_anomalies,
            }
            scores_df = pd.DataFrame(df_data)
            csv_save_path = plot_dir / f"scores_{args.scoring_method}.csv"
            scores_df.to_csv(csv_save_path, index=False)
            print(f"Scores saved to {csv_save_path}")

        metrics = compute_all_metrics(ground_truth_np, predicted_anomalies, scores_for_metrics)
        all_results[set_name] = metrics

        total_tp += metrics["TP"]
        total_tn += metrics["TN"]
        total_fp += metrics["FP"]
        total_fn += metrics["FN"]
        all_ground_truth.append(ground_truth_np)
        all_final_scores.append(np.asarray(scores_for_metrics))

        print("Computed Metrics:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")

    if all_results:
        print("\n--- Computing Overall Metrics ---")
        overall_metrics = compute_metrics_from_cm(total_tp, total_tn, total_fp, total_fn)
        overall_gt = np.concatenate(all_ground_truth)
        overall_scores = np.concatenate(all_final_scores)
        overall_metrics["ROC_AUC"] = calculate_roc_auc(overall_gt, overall_scores)
        all_results["overall"] = overall_metrics

        print("Overall Metrics:")
        for metric_name, metric_value in overall_metrics.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_output_path = output_dir / "detection_metrics.json"
        with open(json_output_path, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=4)
        print(f"\nAll results saved to {json_output_path}")

    return all_results

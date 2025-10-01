import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
from glob import glob
from scipy.ndimage import uniform_filter1d

from ebm_model_vit import EBViTModelWrapper as EBM
from metrics import compute_all_metrics
from utils.utils_visualization import plot_energy_with_anomalies
from anomaly_scoring import detect_with_ema, detect_with_cusum
from img_transformations import WAVEmbedder # For on-the-fly conversion

def reconstruct_scores_from_overlapping_chunks(scores, signal_len, chunk_width, stride, patch_size, image_height, agg_method='max'):
    """
    Reconstructs a single 1D score timeline from scores of overlapping chunks.

    Args:
        scores (np.ndarray): The raw per-patch scores from the model. Shape: (num_chunks, num_patches_per_chunk).
        signal_len (int): The length of the original signal.
        chunk_width (int): The width of each chunk (in time steps).
        stride (int): The stride of the sliding window.
        patch_size (tuple or int): The size of a patch (height, width). If int, assumes square.
        image_height (int): The height of the chunk image (e.g., number of wavelet scales).
        agg_method (str): How to aggregate scores for overlapping patches ('max' or 'mean').

    Returns:
        np.ndarray: A 1D array of anomaly scores for the portion of the signal covered by the chunks.
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    patch_height, patch_width = patch_size

    patches_per_chunk_h = image_height // patch_height
    patches_per_chunk_w = chunk_width // patch_width
    num_signal_patches = signal_len // patch_width

    if agg_method == 'max':
        score_agg = np.full(num_signal_patches, -np.inf, dtype=np.float32)
    else:  # for 'mean'
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
                else:  # 'mean'
                    score_sum[global_patch_idx] += time_patch_scores[j]
                    score_count[global_patch_idx] += 1
                covered_patches[global_patch_idx] = True

    if agg_method == 'max':
        final_scores = np.where(covered_patches, score_agg, 0)
    else:  # 'mean'
        final_scores = np.divide(score_sum, score_count, out=np.zeros_like(score_sum), where=score_count != 0)

    if np.any(covered_patches):
        last_covered_idx = np.where(covered_patches)[0][-1]
    else:
        return np.array([])

    return final_scores[:last_covered_idx + 1]


def detect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model ---
    print(f"Loading checkpoint from {args.ckpt_path}...")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.ckpt_path}")
        
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt['args']
    
    val_chunks_for_shape = torch.load(args.val_data_path)
    if isinstance(val_chunks_for_shape, dict):
        val_chunks_for_shape = val_chunks_for_shape['chunks']
    _B, C, H, W = val_chunks_for_shape.shape

    model = EBM(
        dim=(C, H, W),
        num_channels=train_args.num_channels, num_res_blocks=train_args.num_res_blocks,
        channel_mult=train_args.channel_mult, attention_resolutions=train_args.attention_resolutions,
        num_heads=train_args.num_heads, num_head_channels=train_args.num_head_channels,
        dropout=train_args.dropout, patch_size=train_args.patch_size,
        embed_dim=train_args.embed_dim, transformer_nheads=train_args.transformer_nheads,
        transformer_nlayers=train_args.transformer_nlayers, output_scale=train_args.output_scale,
        energy_clamp=train_args.energy_clamp,
    ).to(device)
    model.load_state_dict(ckpt['ema_model'])
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Setup for Scoring ---
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

            scores_per_patch = model.potential(scores_input, t=torch.ones(scores_input.size(0), device=device))
            val_scores.append(scores_per_patch.mean(dim=1).cpu())
        
        val_scores = torch.cat(val_scores, dim=0).numpy()
    
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
        _, cusum_values = detect_with_cusum(val_scores, baseline=cusum_baseline, k=args.cusum_k)
        anomaly_threshold = np.percentile(cusum_values, args.threshold_percentile)
        print(f"CUSUM baseline set to {cusum_baseline:.4f}, threshold (h) set to {anomaly_threshold:.4f}")

    # --- 4. Evaluate on Preprocessed Test Sets ---
    test_files = sorted(glob(os.path.join(args.test_data_dir, "test_*.pt")))
    test_files = [f for f in test_files if ('_mag' in Path(f).name) == is_mag_only and '_stft' not in Path(f).name]

    if not test_files:
        print(f"No preprocessed test files found in {args.test_data_dir} matching the model type.")
        return

    for test_file_path in test_files:
        set_name = Path(test_file_path).stem
        print(f"\n--- Evaluating {set_name} ---")
        
        data = torch.load(test_file_path, weights_only=False)
        ground_truth = data['labels']
        signal_len = data['signal_len']
        stride = data['stride']

        chunks_to_process = data['raw_chunks'] if args.use_residual_scoring else data['chunks']
        if chunks_to_process.shape[0] == 0:
            print("No chunks in this file, skipping.")
            continue

        all_test_scores_per_patch = []
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

                scores_per_patch = model.potential(scores_input, t=torch.ones(scores_input.size(0), device=device))
                all_test_scores_per_patch.append(scores_per_patch)
            
            test_scores_per_patch = torch.cat(all_test_scores_per_patch, dim=0).cpu().numpy()

        final_scores = reconstruct_scores_from_overlapping_chunks(
            scores=test_scores_per_patch, signal_len=signal_len, chunk_width=W,
            stride=stride, patch_size=train_args.patch_size, image_height=H,
            agg_method=args.overlap_agg_method
        )

        common_len = min(len(final_scores), len(ground_truth))
        final_scores = final_scores[:common_len]
        ground_truth_np = ground_truth[:common_len].numpy()

        if args.scoring_method == 'threshold':
            predicted_anomalies = (final_scores > anomaly_threshold).astype(int)
            plot_title = f"Energy Scores for {set_name}"
            scores_to_plot = final_scores
        elif args.scoring_method == 'ema':
            alarms, ema_values = detect_with_ema(final_scores, alpha=args.ema_alpha, threshold=anomaly_threshold)
            predicted_anomalies = np.array(alarms).astype(int)
            plot_title = f"EMA({args.ema_alpha}) Scores for {set_name}"
            scores_to_plot = ema_values
        elif args.scoring_method == 'cusum':
            alarms, cusum_values = detect_with_cusum(final_scores, baseline=cusum_baseline, h=anomaly_threshold, k=args.cusum_k)
            predicted_anomalies = np.array(alarms).astype(int)
            plot_title = f"CUSUM (h={anomaly_threshold:.2f}) Scores for {set_name}"
            scores_to_plot = cusum_values

        plot_dir = Path("results/plots") / set_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"score_plot_{args.scoring_method}.png"

        plot_energy_with_anomalies(
            energy_scores=scores_to_plot, threshold=anomaly_threshold, save_path=plot_path,
            title=plot_title, ground_truth_labels=ground_truth_np,
        )
        print(f"Scoring plot saved to {plot_path}")
        
        metrics = compute_all_metrics(ground_truth_np, predicted_anomalies, final_scores)
        
        print("Computed Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an EBM model for anomaly detection on preprocessed data.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--val_data_path", type=str, default="preprocessed_dataset/val_chunks_wavelet_mag.pt", help="Path to the validation data for setting the anomaly threshold.")
    parser.add_argument("--test_data_dir", type=str, default="preprocessed_dataset", help="Directory containing the preprocessed test set files (*.pt).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing data to avoid OOM errors.")
    
    parser.add_argument("--scoring_method", type=str, default="threshold", choices=["threshold", "ema", "cusum"], help="The scoring method to use for anomaly detection.")
    parser.add_argument("--threshold_percentile", type=float, default=99.5, help="Percentile of validation scores to use as anomaly threshold.")
    parser.add_argument("--overlap_agg_method", type=str, default="max", choices=["max", "mean"], help="How to aggregate scores from overlapping chunks.")

    # Residual Scoring arguments
    parser.add_argument("--use_residual_scoring", action="store_true", help="If set, perform detrending and score the residual signal.")
    parser.add_argument("--detrend_window_size", type=int, default=256, help="Window size for the moving average filter for detrending.")
    parser.add_argument("--wavelet_name", type=str, default="morl", help="Name of the wavelet to use for on-the-fly CWT.")
    parser.add_argument("--wavelet_scales_min", type=int, default=1, help="Minimum scale for on-the-fly CWT.")
    parser.add_argument("--wavelet_scales_max", type=int, default=129, help="Maximum scale for on-the-fly CWT.")

    # EMA-specific arguments
    parser.add_argument("--ema_alpha", type=float, default=0.1, help="Alpha smoothing factor for EMA scoring.")

    # CUSUM-specific arguments
    parser.add_argument("--cusum_k", type=float, default=0.2, help="Slack parameter (k) for CUSUM scoring.")

    args = parser.parse_args()
    detect(args)
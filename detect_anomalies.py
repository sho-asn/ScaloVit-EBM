"""CLI wrapper for running the detection pipeline."""

import argparse

from scalovit.detection import detect


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an EBM model for anomaly detection on preprocessed data.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--val_data_path", type=str, default="preprocessed_dataset/val_chunks_wavelet_mag.pt", help="Path to the validation data for setting the anomaly threshold.")
    parser.add_argument("--test_data_dir", type=str, default="preprocessed_dataset", help="Directory containing the preprocessed test set files (*.pt).")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing data to avoid OOM errors.")
    parser.add_argument("--scoring_method", type=str, default="threshold", choices=["threshold", "ema", "cusum"], help="The scoring method to use for anomaly detection.")
    parser.add_argument("--threshold_percentile", type=float, default=99.5, help="Percentile of validation scores to use as anomaly threshold.")
    parser.add_argument("--overlap_agg_method", type=str, default="max", choices=["max", "mean"], help="How to aggregate scores from overlapping chunks.")
    parser.add_argument("--use_residual_scoring", action="store_true", help="If set, perform detrending and score the residual signal.")
    parser.add_argument("--detrend_window_size", type=int, default=256, help="Window size for the moving average filter for detrending.")
    parser.add_argument("--wavelet_name", type=str, default="morl", help="Name of the wavelet to use for on-the-fly CWT.")
    parser.add_argument("--wavelet_scales_min", type=int, default=1, help="Minimum scale for on-the-fly CWT.")
    parser.add_argument("--wavelet_scales_max", type=int, default=129, help="Maximum scale for on-the-fly CWT.")
    parser.add_argument("--ema_alpha", type=float, default=0.1, help="Alpha smoothing factor for EMA scoring.")
    parser.add_argument("--cusum_k", type=float, default=0.2, help="Slack parameter (k) for CUSUM scoring.")
    parser.add_argument("--output_dir", type=str, default="results/detection_run", help="Directory to save all outputs (plots, scores, and metrics).")
    parser.add_argument("--save_scores", action="store_true", help="If set, save the computed scores and labels to a CSV file for each test set.")

    args = parser.parse_args()
    detect(args)
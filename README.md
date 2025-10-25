# ScaloVit-EBM Repository

This repository accompanies the paper **“ScaloVit-EBM: Localized Energy-Based Anomaly Detection on Time–Frequency Scalograms”**. It contains the code needed to preprocess the Cranfield three-phase flow dataset, train the ScaloVit-EBM model, and run the evaluation pipelines described in the manuscript.

## Repository Layout

- `scalovit/`: Python package containing models, data preprocessing helpers, evaluation utilities, and transforms used in the paper.
- `preprocess_data.py`: CLI wrapper that calls `scalovit.data.preprocessing.preprocess`.
- `train_ebm.py`, `train_ebm_ablation.py`: training entry points for the main model and ablation variants (using `scalovit.training.loop`).
- `detect_anomalies.py`, `detect_anomalies_ablation.py`: evaluation entry points backed by `scalovit.detection.pipeline`.
- `requirements.txt`: Python dependency specification.
- `README.md`: this file.

## Setup

1. Create and activate a Python environment (Python ≥ 3.10).
2. Install dependencies via `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Data Preparation

1. Download the Cranfield three-phase flow facility dataset (CVACaseStudy/MFP).
2. Place the `.mat` files under `Datasets/CVACaseStudy/MFP/` (create the directory locally; it is not tracked in this repository).
3. Run preprocessing to detrend signals, generate CWT scalograms, and save chunked tensors. Example command (adjust paths/flags as needed):

```bash
python preprocess_data.py \
  --transform_type wavelet \
  --data_dir Datasets/CVACaseStudy/MFP \
  --output_dir preprocessed_dataset \
  --chunk_width 2048 \
  --chunk_stride 64 \
  --detrend --detrend_window_size 4096
```

## Training & Evaluation

Train the primary ScaloVit-EBM model (Energy Matching, FM-only). If you used `--detrend`, the files include `_residual` in their names:

```bash
python train_ebm.py \
	--train_data_path preprocessed_dataset/train_chunks_wavelet_residual_mag.pt \
	--val_data_path preprocessed_dataset/val_chunks_wavelet_residual_mag.pt \
	--output_dir results/ebm_fm
```

Evaluate on the test sets and compute metrics (note the residual filenames if detrending was enabled):

```bash
python detect_anomalies.py \
	--ckpt_path path/to/checkpoint.pt \
	--val_data_path preprocessed_dataset/val_chunks_wavelet_residual_mag.pt \
	--test_data_dir preprocessed_dataset \
	--overlap_agg_method max \
	--threshold_percentile 99.5 \
	--output_dir results/ebm_detection
```

To reproduce ablations (Contrastive Divergence, alternative heads, aggregation strategies), use the corresponding scripts:

```bash
python train_ebm_ablation.py --lambda_cd 0.001 --output_dir results/ebm_cd
python detect_anomalies_ablation.py --ckpt_path path/to/cd_checkpoint.pt --output_dir results/ebm_cd_detection
```

Per-signal scoring and metrics utilities are exposed through `scalovit.scoring` and `scalovit.metrics`.

## Best model quickstart (paper settings)

These flags match the best-performing configuration reported in the paper (FM-only; no CD), with detrended magnitude CWT inputs and max aggregation:

```bash
# Training
python train_ebm.py \
	--total_steps 10000 \
	--warmup 5000 \
	--save_step 300 \
	--lr 0.001 \
	--batch_size 4 \
	--gradient_accumulation_steps 2 \
	--num_channels 64 \
	--embed_dim 192 \
	--transformer_nheads 8 \
	--transformer_nlayers 8 \
	--num_heads 4 \
	--num_head_channels 64 \
	--n_gibbs 0 \
	--epsilon_max 0.0 \
	--lambda_cd 0.0 \
	--time_cutoff 1.0 \
	--use_amp \
	--train_data_path preprocessed_dataset/train_chunks_wavelet_residual_mag.pt \
	--val_data_path preprocessed_dataset/val_chunks_wavelet_residual_mag.pt \
	--output_dir results/ebm_best

# Detection (threshold from normal validation; max overlap aggregation)
python detect_anomalies.py \
	--ckpt_path results/ebm_best/EBM_residual_vit_step_2700.pt \
	--val_data_path preprocessed_dataset/val_chunks_wavelet_residual_mag.pt \
	--test_data_dir preprocessed_dataset \
	--batch_size 4 \
	--overlap_agg_method max \
	--threshold_percentile 99.5 \
	--output_dir results/ebm_best_detection
```

Chunking/representation used in preprocessing:

- chunk_width=2048, chunk_stride=64
- detrend_window_size=4096, wavelet_scales_max=129
- patch_size=[128, 8] (frequency × time), magnitude-only

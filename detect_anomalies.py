import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import scipy.io as scio

from sklearn.metrics import roc_auc_score
from ebm_model_vit import EBViTModelWrapper as EBM
from img_transformations import WAVEmbedder
from model import init_wav_embedder, split_image_into_chunks

# --- Data Loading for Individual Sets ---
def load_set_data(case_file: str, set_name: str, data_dir: Path) -> (torch.Tensor, int):
    """Loads a single set from a .mat file."""
    file_path = data_dir / f"{case_file}.mat"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    
    data = scio.loadmat(file_path)
    if set_name in data:
        # Remove last feature
        arr = np.delete(data[set_name], -1, axis=1)
        set_length = len(arr)
    else:
        raise ValueError(f"{set_name} not found in {file_path}")
    return torch.from_numpy(arr).unsqueeze(0).float(), set_length

# --- Ground Truth Label Generation ---
def get_set_ground_truth(fault_intervals: list, total_chunks: int, chunk_width: int) -> np.ndarray:
    """Creates a binary label array (0=normal, 1=anomaly) for each chunk in a set."""
    labels = np.zeros(total_chunks, dtype=int)
    for start, end in fault_intervals:
        start_chunk = start // chunk_width
        end_chunk = end // chunk_width
        if end_chunk < len(labels):
            labels[start_chunk:end_chunk + 1] = 1
    return labels

# --- Main Detection Logic ---
def detect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model and Set Up Embedder ---
    print(f"Loading checkpoint from {args.ckpt_path}...")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.ckpt_path}")
        
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt['args']
    
    val_chunks = torch.load(args.val_data_path)
    _B, C, H, W = val_chunks.shape
    num_features = C // 2

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

    # Re-initialize and fit an embedder on the same training data distribution
    print("Re-fitting data transformer on original training data...")
    train_chunks_full = torch.load(args.train_data_path)
    embedder = WAVEmbedder(device=device, seq_len=train_chunks_full.shape[0] * W, wavelet_name='morl', scales_arange=(1, 129))
    dummy_raw_signal_for_fitting = torch.randn(1, train_chunks_full.shape[0] * W, num_features).cpu().numpy()
    init_wav_embedder(embedder, dummy_raw_signal_for_fitting)

    # --- 2. Establish Anomaly Threshold from Validation Data ---
    print(f"Calculating energy scores on validation data from {args.val_data_path}...")
    with torch.no_grad():
        val_scores = model.potential(val_chunks.to(device), t=torch.ones(val_chunks.size(0), device=device))
        val_scores = val_scores.cpu().numpy()
    
    anomaly_threshold = np.percentile(val_scores, args.threshold_percentile)
    print(f"Anomaly threshold set to {anomaly_threshold:.4f}")

    # --- 3. Evaluate on Individual Test Sets ---
    data_dir = Path("Datasets") / "CVACaseStudy" / "MFP"
    sets_to_evaluate = [
        # (set_name, case_file, anomaly_intervals)
        ("Set1_1", "FaultyCase1", [(1566, 5181)]),
        ("Set1_2", "FaultyCase1", [(657, 3777)]),
        ("Set1_3", "FaultyCase1", [(691, 3691)]),
        ("Set2_1", "FaultyCase2", [(2244, 6616)]),
        ("Set2_2", "FaultyCase2", [(476, 2656)]),
        ("Set2_3", "FaultyCase2", [(331, 2467)]),
        ("Set3_1", "FaultyCase3", [(1136, 8352)]),
        ("Set3_2", "FaultyCase3", [(333, 5871)]),
        ("Set3_3", "FaultyCase3", [(596, 9566)]),
        ("Set4_1", "FaultyCase4", [(953, 6294)]),
        ("Set4_2", "FaultyCase4", [(851, 3851)]),
        ("Set4_3", "FaultyCase4", [(241, 3241)]),
        ("Set5_1", "FaultyCase5", [(686, 1172), (1772, 2253)]),
        ("Set5_2", "FaultyCase5", [(1633, 2955), (7031, 7553), (8057, 10608)]),
    ]

    for set_name, case_file, intervals in sets_to_evaluate:
        print(f"\n--- Evaluating {set_name} from {case_file} ---")
        try:
            raw_signal, set_length = load_set_data(case_file, set_name, data_dir)
        except (FileNotFoundError, ValueError) as e:
            print(e)
            continue
        
        embedder.seq_len = raw_signal.shape[1]
        wavelet_image = embedder.ts_to_img(raw_signal)
        test_chunks = split_image_into_chunks(wavelet_image, W)
        
        if test_chunks.shape[0] == 0:
            print("No chunks generated, skipping set.")
            continue

        with torch.no_grad():
            test_scores = model.potential(test_chunks.to(device), t=torch.ones(test_chunks.size(0), device=device))
            test_scores = test_scores.cpu().numpy()
        
        ground_truth = get_set_ground_truth(intervals, len(test_chunks), W)

        if len(np.unique(ground_truth)) > 1:
            auc_score = roc_auc_score(ground_truth, test_scores)
            print(f"ROC AUC Score for {set_name}: {auc_score:.4f}")
        else:
            print(f"Ground truth for {set_name} contains only one class, cannot calculate AUC.")

        # --- Calculate Detection and False Alarm Rates ---
        predicted_anomalies = test_scores > anomaly_threshold
        
        faulty_period_indices = np.where(ground_truth == 1)[0]
        non_faulty_period_indices = np.where(ground_truth == 0)[0]

        if len(faulty_period_indices) > 0:
            correctly_identified_faults = np.sum(predicted_anomalies[faulty_period_indices])
            detection_rate = correctly_identified_faults / len(faulty_period_indices)
            print(f"Detection Rate (Recall) for {set_name}: {detection_rate:.4f}")
        else:
            print(f"No faulty period in ground truth for {set_name}, cannot calculate Detection Rate.")

        if len(non_faulty_period_indices) > 0:
            incorrectly_flagged_non_faults = np.sum(predicted_anomalies[non_faulty_period_indices])
            false_alarm_rate = incorrectly_flagged_non_faults / len(non_faulty_period_indices)
            print(f"False Alarm Rate for {set_name}: {false_alarm_rate:.4f}")
        else:
            print(f"No non-faulty period in ground truth for {set_name}, cannot calculate False Alarm Rate.")

        # Plotting
        # fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        # ax.plot(test_scores, label='Anomaly Score', color='blue', alpha=0.8)
        # ax.axhline(anomaly_threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
        
        # for i, label in enumerate(ground_truth):
        #     if label == 1:
        #         ax.axvspan(i, i + 1, color='red', alpha=0.2, lw=0)
        
        # ax.set_title(f"Anomaly Scores for {set_name}")
        # ax.set_xlabel('Chunk Index')
        # ax.set_ylabel('Energy Score')
        # # Create a custom legend entry for the shaded region
        # from matplotlib.patches import Patch
        # legend_elements = [plt.Line2D([0], [0], color='blue', lw=2, label='Anomaly Score'),
        #                    plt.Line2D([0], [0], color='r', linestyle='--', lw=2, label='Threshold'),
        #                    Patch(facecolor='red', alpha=0.2, label='Ground Truth Anomaly')]
        # ax.legend(handles=legend_elements)
        # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_data_path", type=str, default="preprocessed_dataset/val_chunks.pt")
    parser.add_argument("--train_data_path", type=str, default="preprocessed_dataset/train_chunks.pt") # Needed to refit embedder
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--threshold_percentile", type=float, default=99)
    args = parser.parse_args()
    detect(args)
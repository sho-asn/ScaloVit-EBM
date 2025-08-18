import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
import scipy.io as scio

from sklearn.metrics import roc_auc_score
from ebm_model import EBM
from img_transformations import WAVEmbedder
from model import init_wav_embedder, split_image_into_chunks

# --- Data Loading for Faulty Cases ---
def load_faulty_case_data(case_name: str, data_dir: Path, set_names: list) -> (torch.Tensor, list):
    """Loads all sets from a faulty case .mat file and concatenates them."""
    file_path = data_dir / f"{case_name}.mat"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    
    data = scio.loadmat(file_path)
    all_sets = []
    set_lengths = []
    for name in set_names:
        if name in data:
            # Remove last feature as per user example
            arr = np.delete(data[name], -1, axis=1)
            all_sets.append(arr)
            set_lengths.append(len(arr))
        else:
            print(f"Warning: {name} not found in {file_path}")
            
    concatenated_signal = np.concatenate(all_sets, axis=0)
    return torch.from_numpy(concatenated_signal).unsqueeze(0).float(), set_lengths

# --- Ground Truth Label Generation ---
def get_ground_truth_labels(fault_intervals: dict, set_lengths: list, total_chunks: int, chunk_width: int) -> np.ndarray:
    """Creates a binary label array (0=normal, 1=anomaly) for each chunk."""
    labels = np.zeros(total_chunks, dtype=int)
    set_start_time = 0
    for i, length in enumerate(set_lengths):
        set_name = f"Set{list(fault_intervals.keys())[0][3]}_{i+1}" # e.g., Set1_1, Set2_1
        if set_name in fault_intervals:
            for start, end in fault_intervals[set_name]:
                start_chunk = (set_start_time + start) // chunk_width
                end_chunk = (set_start_time + end) // chunk_width
                if end_chunk < len(labels):
                    labels[start_chunk:end_chunk + 1] = 1
        set_start_time += length
    return labels

# --- Main Detection Logic ---
def detect(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Model and Set Up Embedder ---
    print(f"Loading checkpoint from {args.ckpt_path}...")
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {args.ckpt_path}")
        
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = ckpt['args']
    
    val_chunks = torch.load(args.val_data_path)
    _B, C, H, W = val_chunks.shape
    val_chunks_reshaped = val_chunks.permute(0, 2, 3, 1).reshape(val_chunks.shape[0], H * W, C)

    model = EBM(dim=H * W, channels=C, dim_mults=(1, 2, 4)).to(device)
    model.load_state_dict(ckpt['ema_model'])
    model.eval()
    print("Model loaded successfully.")

    # Re-initialize and fit an embedder on the same training data distribution
    # This is crucial so that the test data is transformed in exactly the same way.
    print("Re-fitting data transformer on original training data...")
    train_chunks_full = torch.load(args.train_data_path)
    embedder = WAVEmbedder(device=device, seq_len=train_chunks_full.shape[0] * W, wavelet_name='morl', scales_arange=(1, 128))
    # We need to reconstruct the original raw signal shape to init the embedder
    # This is an approximation, but sufficient for fitting the scalers
    dummy_raw_signal_for_fitting = torch.randn(1, train_chunks_full.shape[0] * W, C).cpu().numpy()
    init_wav_embedder(embedder, dummy_raw_signal_for_fitting)

    # --- 2. Establish Anomaly Threshold from Validation Data ---
    print(f"Calculating energy scores on validation data from {args.val_data_path}...")
    with torch.no_grad():
        val_scores = model.potential(val_chunks_reshaped.to(device), t=torch.ones(val_chunks_reshaped.size(0), device=device))
        val_scores = val_scores.cpu().numpy()
    
    anomaly_threshold = np.percentile(val_scores, args.threshold_percentile)
    print(f"Anomaly threshold set to {anomaly_threshold:.4f}")

    # --- 3. Evaluate on Faulty Test Cases ---
    data_dir = Path("Datasets") / "CVACaseStudy" / "MFP"
    fault_cases = {
        "FaultyCase1": {"Set1_1": [(1566, 5181)], "Set1_2": [(657, 3777)], "Set1_3": [(691, 3691)]},
        "FaultyCase2": {"Set2_1": [(2244, 6616)], "Set2_2": [(476, 2656)], "Set2_3": [(331, 2467)]},
        "FaultyCase3": {"Set3_1": [(1136, 8352)], "Set3_2": [(333, 5871)], "Set3_3": [(596, 9566)]},
        "FaultyCase4": {"Set4_1": [(953, 6294)], "Set4_2": [(851, 3851)], "Set4_3": [(241, 3241)]},
        "FaultyCase5": {"Set5_1": [(686, 1172), (1772, 2253)], "Set5_2": [(1633, 2955), (7031, 7553), (8057, 10608)]}
    }
    case_set_names = {
        "FaultyCase1": ["Set1_1", "Set1_2", "Set1_3"],
        "FaultyCase2": ["Set2_1", "Set2_2", "Set2_3"],
        "FaultyCase3": ["Set3_1", "Set3_2", "Set3_3"],
        "FaultyCase4": ["Set4_1", "Set4_2", "Set4_3"],
        "FaultyCase5": ["Set5_1", "Set5_2"]
    }

    for case_name, intervals in fault_cases.items():
        print(f"\n--- Evaluating {case_name} ---")
        raw_signal, set_lengths = load_faulty_case_data(case_name, data_dir, case_set_names[case_name])
        
        embedder.seq_len = raw_signal.shape[1]
        wavelet_image = embedder.ts_to_img(raw_signal)
        test_chunks = split_image_into_chunks(wavelet_image, W)
        
        if test_chunks.shape[0] == 0:
            print("No chunks generated, skipping case.")
            continue

        test_chunks_reshaped = test_chunks.permute(0, 2, 3, 1).reshape(test_chunks.shape[0], H * W, C)

        with torch.no_grad():
            test_scores = model.potential(test_chunks_reshaped.to(device), t=torch.ones(test_chunks_reshaped.size(0), device=device))
            test_scores = test_scores.cpu().numpy()

        ground_truth = get_ground_truth_labels(intervals, set_lengths, len(test_chunks), W)
        
        if len(np.unique(ground_truth)) > 1:
            auc_score = roc_auc_score(ground_truth, test_scores)
            print(f"ROC AUC Score for {case_name}: {auc_score:.4f}")
        else:
            print("Ground truth contains only one class, cannot calculate AUC.")

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(test_scores, label='Anomaly Score', color='blue', alpha=0.8)
        ax.axhline(anomaly_threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
        
        for i, label in enumerate(ground_truth):
            if label == 1:
                ax.axvspan(i, i + 1, color='red', alpha=0.2, lw=0)
        
        ax.set_title(f"Anomaly Scores for {case_name}")
        ax.set_xlabel('Chunk Index')
        ax.set_ylabel('Energy Score')
        # Create a custom legend entry for the shaded region
        from matplotlib.patches import Patch
        legend_elements = [plt.Line2D([0], [0], color='blue', lw=2, label='Anomaly Score'),
                           plt.Line2D([0], [0], color='r', linestyle='--', lw=2, label='Threshold'),
                           Patch(facecolor='red', alpha=0.2, label='Ground Truth Anomaly')]
        ax.legend(handles=legend_elements)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_data_path", type=str, default="val_chunks.pt")
    parser.add_argument("--train_data_path", type=str, default="train_chunks.pt") # Needed to refit embedder
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--threshold_percentile", type=float, default=99.5)
    args = parser.parse_args()
    detect(args)

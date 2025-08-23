import torch
import numpy as np
from pathlib import Path
import scipy.io as scio
import os

from img_transformations import WAVEmbedder, init_wav_embedder, split_image_into_chunks

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_WIDTH = 128
TRAIN_SPLIT_RATIO = 0.8
DATA_DIR = Path("Datasets") / "CVACaseStudy" / "MFP"
OUTPUT_DIR = Path("preprocessed_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

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

# --- Main Preprocessing Logic ---
def preprocess():
    print(f"Using device: {device}")

    # --- 1. Load and Prepare Training Data for Fitting the Embedder ---
    print("--- Preparing Training Data ---")
    training_data_path = DATA_DIR / "Training.mat"
    data = scio.loadmat(training_data_path)
    t1_raw = np.delete(data['T1'], -1, axis=1)
    t2_raw = np.delete(data['T2'], -1, axis=1)
    t3_raw = np.delete(data['T3'], -1, axis=1)
    print(f"Loaded T1, T2, T3 with shapes: {t1_raw.shape}, {t2_raw.shape}, {t3_raw.shape}")

    all_train_raw = []
    all_val_raw = []
    for signal_np in [t1_raw, t2_raw, t3_raw]:
        split_point = int(signal_np.shape[0] * TRAIN_SPLIT_RATIO)
        all_train_raw.append(signal_np[:split_point, :])
        all_val_raw.append(signal_np[split_point:, :])

    # --- 2. Initialize and Fit Embedder on COMBINED TRAINING DATA ---
    combined_train_raw_np = np.concatenate(all_train_raw, axis=0)
    print(f"Combined raw training data to fit scaler, shape: {combined_train_raw_np.shape}")

    embedder = WAVEmbedder(
        device=device,
        seq_len=combined_train_raw_np.shape[0], # This will be dynamically set later
        wavelet_name='morl',
        scales_arange=(1, 129)
    )

    print("Caching normalization parameters from all training data...")
    # The first argument to init_wav_embedder needs to be a (1, L, F) array
    init_wav_embedder(embedder, np.expand_dims(combined_train_raw_np, axis=0))
    print("Normalization parameters cached.")

    # --- 3. Save Normalization Parameters ---
    norm_params = {
        'min_mag': embedder.min_mag.cpu(),
        'max_mag': embedder.max_mag.cpu(),
        'min_phase': embedder.min_phase.cpu(),
        'max_phase': embedder.max_phase.cpu(),
    }
    norm_params_path = OUTPUT_DIR / "norm_params.pt"
    torch.save(norm_params, norm_params_path)
    print(f"Saved normalization parameters to {norm_params_path}")

    # --- 4. Transform and Save Train/Val Chunks ---
    print("--- Processing and Saving Train/Val Sets ---")
    final_train_chunks = []
    for train_set_np in all_train_raw:
        train_tensor = torch.from_numpy(np.expand_dims(train_set_np, axis=0)).float().to(device)
        embedder.seq_len = train_tensor.shape[1]
        wavelet_image = embedder.ts_to_img(train_tensor)
        chunks = split_image_into_chunks(wavelet_image, CHUNK_WIDTH)
        final_train_chunks.append(chunks)
    final_train_chunks = torch.cat(final_train_chunks, dim=0)
    torch.save(final_train_chunks, OUTPUT_DIR / "train_chunks.pt")
    print(f"Saved {final_train_chunks.shape} total training chunks.")

    final_val_chunks = []
    for val_set_np in all_val_raw:
        val_tensor = torch.from_numpy(np.expand_dims(val_set_np, axis=0)).float().to(device)
        embedder.seq_len = val_tensor.shape[1]
        wavelet_image = embedder.ts_to_img(val_tensor)
        chunks = split_image_into_chunks(wavelet_image, CHUNK_WIDTH)
        final_val_chunks.append(chunks)
    final_val_chunks = torch.cat(final_val_chunks, dim=0)
    torch.save(final_val_chunks.float(), OUTPUT_DIR / "val_chunks.pt")
    print(f"Saved {final_val_chunks.shape} total validation chunks.")

    # --- 5. Process and Save Test Sets ---
    print("--- Processing and Saving Test Sets ---")
    sets_to_evaluate = [
        ("Set1_1", "FaultyCase1", [(1566, 5181)]), ("Set1_2", "FaultyCase1", [(657, 3777)]),
        ("Set1_3", "FaultyCase1", [(691, 3691)]), ("Set2_1", "FaultyCase2", [(2244, 6616)]),
        ("Set2_2", "FaultyCase2", [(476, 2656)]), ("Set2_3", "FaultyCase2", [(331, 2467)]),
        ("Set3_1", "FaultyCase3", [(1136, 8352)]), ("Set3_2", "FaultyCase3", [(333, 5871)]),
        ("Set3_3", "FaultyCase3", [(596, 9566)]), ("Set4_1", "FaultyCase4", [(953, 6294)]),
        ("Set4_2", "FaultyCase4", [(851, 3851)]), ("Set4_3", "FaultyCase4", [(241, 3241)]),
        ("Set5_1", "FaultyCase5", [(686, 1172), (1772, 2253)]),
        ("Set5_2", "FaultyCase5", [(1633, 2955), (7031, 7553), (8057, 10608)]),
    ]

    for set_name, case_file, intervals in sets_to_evaluate:
        print(f"Processing {set_name} from {case_file}...")
        try:
            file_path = DATA_DIR / f"{case_file}.mat"
            data = scio.loadmat(file_path)
            raw_signal_np = np.delete(data[set_name], -1, axis=1)
            raw_signal = torch.from_numpy(np.expand_dims(raw_signal_np, axis=0)).float().to(device)

            embedder.seq_len = raw_signal.shape[1]
            wavelet_image = embedder.ts_to_img(raw_signal)
            test_chunks = split_image_into_chunks(wavelet_image, CHUNK_WIDTH)

            if test_chunks.shape[0] == 0:
                print(f"No chunks generated for {set_name}, skipping.")
                continue

            ground_truth = get_set_ground_truth(intervals, len(test_chunks), CHUNK_WIDTH)

            save_path = OUTPUT_DIR / f"test_{case_file}_{set_name}.pt"
            torch.save({'chunks': test_chunks.cpu().float(), 'labels': ground_truth}, save_path)
            print(f"Saved {test_chunks.shape[0]} chunks and labels to {save_path}")

        except (FileNotFoundError, KeyError) as e:
            print(f"Could not process {set_name} from {case_file}. Reason: {e}")

    print("\nData preparation complete.")

if __name__ == "__main__":
    preprocess()

import torch
import numpy as np
from pathlib import Path
import scipy.io as scio
import os
import argparse

from img_transformations import (
    WAVEmbedder, init_wav_embedder, 
    STFTEmbedder, init_stft_embedder,
    split_image_into_chunks_with_stride
)
from utils.utils_visualization import plot_stft_images

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess time series data into image chunks.")
    parser.add_argument("--transform_type", type=str, default="wavelet", choices=["wavelet", "stft"], help="Type of transform to use.")
    parser.add_argument("--data_dir", type=str, default="Datasets/CVACaseStudy/MFP", help="Directory of the raw .mat files.")
    parser.add_argument("--output_dir", type=str, default="preprocessed_dataset", help="Directory to save the processed files.")
    parser.add_argument("--chunk_width", type=int, default=2048, help="Width of the output image chunks.")
    parser.add_argument("--chunk_stride", type=int, default=128, help="Stride for sliding window chunking.")
    parser.add_argument("--patch_size", type=int, nargs=2, default=[128, 16], help="Patch size (height, width) used by the model, for labeling.")
    parser.add_argument("--train_split_ratio", type=float, default=0.8, help="Ratio of data to use for training.")
    parser.add_argument("--include_phase", action="store_true", help="If set, include phase in the output image. Otherwise, only magnitude is used.")

    # Wavelet specific args
    parser.add_argument("--wavelet_name", type=str, default="morl", help="Name of the wavelet to use.")
    parser.add_argument("--wavelet_scales_min", type=int, default=1, help="Minimum scale for wavelet transform.")
    parser.add_argument("--wavelet_scales_max", type=int, default=257, help="Maximum scale for wavelet transform.")

    # STFT specific args
    parser.add_argument("--stft_nperseg", type=int, default=62, help="Length of each segment for STFT.")
    parser.add_argument("--stft_noverlap", type=int, default=31, help="Number of points to overlap between segments for STFT.")
    parser.add_argument("--stft_nfft", type=int, default=62, help="Length of the FFT used for STFT.")

    return parser.parse_args()


# --- Ground Truth Label Generation ---
def get_ground_truth_for_signal(fault_intervals: list, signal_length: int, patch_width: int) -> torch.Tensor:
    """
    Creates a single 1D binary label tensor for the entire signal, where each element
    represents a time window of `patch_width`.
    """
    # 1. Create a high-resolution ground truth vector for the entire signal (per time step)
    ts_labels = np.zeros(signal_length, dtype=int)
    for start, end in fault_intervals:
        start = max(0, start)
        end = min(signal_length, end)
        if start < end:
            ts_labels[start:end] = 1

    # 2. Create patch-level labels by checking each time window
    num_patches = signal_length // patch_width
    patch_labels = np.zeros(num_patches, dtype=int)
    for i in range(num_patches):
        start_time = i * patch_width
        end_time = start_time + patch_width
        if np.any(ts_labels[start_time:end_time] == 1):
            patch_labels[i] = 1
            
    return torch.from_numpy(patch_labels).long()


# --- Helper Functions for Preprocessing ---
def chunk_1d_signal_with_stride(signal: np.ndarray, chunk_width: int, stride: int) -> np.ndarray:
    """
    Splits a 2D signal (L, F) into 3D chunks (num_chunks, chunk_width, F) using a sliding window.
    """
    L, F = signal.shape
    chunks = []
    for i in range(0, L - chunk_width + 1, stride):
        chunks.append(signal[i:i+chunk_width, :])
    if not chunks:
        return np.empty((0, chunk_width, F), dtype=signal.dtype)
    return np.stack(chunks, axis=0)

def transform_and_chunk_signal(
    signal_np: np.ndarray,
    embedder,
    args,
    device
) -> torch.Tensor:
    """
    Transforms a single raw signal numpy array to an image and splits it into chunks.
    """
    signal_tensor = torch.from_numpy(np.expand_dims(signal_np, axis=0)).float().to(device)
    embedder.seq_len = signal_tensor.shape[1]
    image = embedder.ts_to_img(signal_tensor)
    
    if not args.include_phase:
        # Keep only magnitude channels (even indices)
        image = image[:, ::2, :, :]

    return split_image_into_chunks_with_stride(image, args.chunk_width, args.chunk_stride)

def process_signal_list(
    raw_signals: list,
    embedder,
    args,
    device,
    set_name: str
) -> torch.Tensor:
    """
    Processes a list of raw signal numpy arrays and returns a concatenated tensor of chunks.
    This version is memory-optimized: it calculates the total size first, pre-allocates
    a CPU tensor, and fills it iteratively to avoid high peak memory usage.
    """
    if not raw_signals or all(s.shape[0] == 0 for s in raw_signals):
        return torch.empty(0)

    # --- Pass 1: Calculate total number of chunks and final tensor shape ---
    print(f"  Calculating size for {set_name} set...")
    total_chunks = 0
    for signal_np in raw_signals:
        if signal_np.shape[0] < args.chunk_width:
            continue
        # Calculate how many chunks this signal will produce
        num_signal_chunks = (signal_np.shape[0] - args.chunk_width) // args.chunk_stride + 1
        total_chunks += num_signal_chunks

    if total_chunks == 0:
        return torch.empty(0)

    # Determine the shape of a single chunk (C, H, W) without processing
    num_features = raw_signals[0].shape[1]
    chunk_w = args.chunk_width
    if args.transform_type == 'wavelet':
        chunk_h = args.wavelet_scales_max - args.wavelet_scales_min
    else: # stft
        chunk_h = args.stft_nfft // 2 + 1
    
    chunk_c = num_features * 2 if args.include_phase else num_features

    final_shape = (chunk_c, chunk_h, chunk_w)

    # --- Pre-allocate memory on CPU ---
    print(f"  Pre-allocating memory for {total_chunks} chunks with shape {final_shape}...")
    all_chunks_cpu = torch.empty((total_chunks, *final_shape), dtype=torch.float32, device='cpu')
    
    # --- Pass 2: Process signals and fill the CPU tensor ---
    current_pos = 0
    for i, signal_np in enumerate(raw_signals):
        print(f"  Processing {set_name} set part {i+1}/{len(raw_signals)}...")
        if signal_np.shape[0] < args.chunk_width:
            print("    Signal shorter than chunk width, skipping.")
            continue

        # Generate chunks on the target device (e.g., GPU)
        with torch.no_grad():
            chunks_device = transform_and_chunk_signal(signal_np, embedder, args, device)
        
        num_generated = chunks_device.shape[0]
        if num_generated > 0:
            print(f"    Generated {num_generated} chunks, moving to CPU memory.")
            # Copy chunks from GPU to the pre-allocated CPU tensor
            all_chunks_cpu[current_pos : current_pos + num_generated] = chunks_device.cpu()
            current_pos += num_generated
            # Free GPU memory
            del chunks_device
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            print("    No chunks generated for this part.")

    # If there was a miscalculation in the number of chunks, truncate the tensor
    if current_pos < total_chunks:
        all_chunks_cpu = all_chunks_cpu[:current_pos]

    return all_chunks_cpu


# --- Main Preprocessing Logic ---
def preprocess(args):
    print(f"Using device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    data_dir = Path(args.data_dir)

    # Define file suffix based on args
    file_suffix = f"_{args.transform_type}"
    if not args.include_phase:
        file_suffix += "_mag"
        print("--- Processing in magnitude-only mode ---")
    else:
        print("--- Processing in magnitude-and-phase mode ---")

    # --- 1. Load and Prepare Training Data for Fitting the Embedder ---
    print("--- Preparing Training Data ---")
    training_data_path = data_dir / "Training.mat"
    data = scio.loadmat(training_data_path)
    t1_raw = np.delete(data['T1'], -1, axis=1)
    t2_raw = np.delete(data['T2'], -1, axis=1)
    t3_raw = np.delete(data['T3'], -1, axis=1)
    print(f"Loaded T1, T2, T3 with shapes: {t1_raw.shape}, {t2_raw.shape}, {t3_raw.shape}")

    all_train_raw = []
    all_val_raw = []
    for signal_np in [t1_raw, t2_raw, t3_raw]:
        split_point = int(signal_np.shape[0] * args.train_split_ratio)
        all_train_raw.append(signal_np[:split_point, :])
        all_val_raw.append(signal_np[split_point:, :])

    # --- 2. Initialize and Fit Embedder on Training Data ---
    # For CWT, we no longer combine all training data into one massive array.
    # The embedder is fitted iteratively on the list of training signals to save memory.
    if args.transform_type == 'wavelet':
        print(f"Found {len(all_train_raw)} training signals to fit CWT scaler.")
        embedder = WAVEmbedder(
            device=device,
            seq_len=0, # seq_len is now set per-signal in transform_and_chunk_signal
            wavelet_name=args.wavelet_name,
            scales_arange=(args.wavelet_scales_min, args.wavelet_scales_max)
        )
        print("Caching wavelet normalization parameters from all training data (iteratively)...")
        init_wav_embedder(embedder, all_train_raw)
    elif args.transform_type == 'stft':
        combined_train_raw_np = np.concatenate(all_train_raw, axis=0)
        print(f"Combined raw training data to fit scaler, shape: {combined_train_raw_np.shape}")
        embedder = STFTEmbedder(
            device=device,
            seq_len=combined_train_raw_np.shape[0],
            nperseg=args.stft_nperseg,
            noverlap=args.stft_noverlap,
            nfft=args.stft_nfft
        )
        print("Caching STFT normalization parameters from all training data...")
        init_stft_embedder(embedder, np.expand_dims(combined_train_raw_np, axis=0))
    else:
        raise ValueError(f"Unknown transform_type: {args.transform_type}")

    print("Normalization parameters cached.")

    # --- 3. Save Normalization Parameters ---
    norm_params = {
        'min_mag': embedder.min_mag.cpu(),
        'max_mag': embedder.max_mag.cpu(),
    }
    if args.include_phase:
        norm_params['min_phase'] = embedder.min_phase.cpu()
        norm_params['max_phase'] = embedder.max_phase.cpu()
    
    norm_params_path = output_dir / f"norm_params{file_suffix}.pt"
    torch.save(norm_params, norm_params_path)
    print(f"Saved normalization parameters to {norm_params_path}")

    # --- 4. Transform and Save Train/Val Chunks ---
    print("--- Processing and Saving Train/Val Sets ---")
    final_train_chunks = process_signal_list(all_train_raw, embedder, args, device, "training")
    torch.save(final_train_chunks, output_dir / f"train_chunks{file_suffix}.pt")
    print(f"Saved {final_train_chunks.shape} total training chunks.")

    if args.transform_type == 'stft':
        plot_dir = Path("results/plots")
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f"stft_training_images{file_suffix}.png"
        plot_stft_images(final_train_chunks, plot_path)

    final_val_chunks = process_signal_list(all_val_raw, embedder, args, device, "validation")
    torch.save(final_val_chunks.float(), output_dir / f"val_chunks{file_suffix}.pt")
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
            file_path = data_dir / f"{case_file}.mat"
            data = scio.loadmat(file_path)
            raw_signal_np = np.delete(data[set_name], -1, axis=1)
            
            signal_len = raw_signal_np.shape[0]
            # Create scalogram chunks (for compatibility if needed)
            test_chunks_scalogram = transform_and_chunk_signal(raw_signal_np, embedder, args, device)
            # Create raw 1D signal chunks
            test_chunks_1d = chunk_1d_signal_with_stride(raw_signal_np, args.chunk_width, args.chunk_stride)

            if test_chunks_scalogram.shape[0] == 0:
                print(f"No chunks generated for {set_name}, skipping.")
                continue

            ground_truth = get_ground_truth_for_signal(
                fault_intervals=intervals,
                signal_length=signal_len,
                patch_width=args.patch_size[1]
            )

            save_path = output_dir / f"test_{case_file}_{set_name}{file_suffix}.pt"
            torch.save({
                'chunks': test_chunks_scalogram.cpu().float(), 
                'raw_chunks': torch.from_numpy(test_chunks_1d).float(),
                'labels': ground_truth,
                'signal_len': signal_len,
                'stride': args.chunk_stride
            }, save_path)
            print(f"Saved {test_chunks_scalogram.shape[0]} chunks and labels of shape {ground_truth.shape} to {save_path}")

        except (FileNotFoundError, KeyError) as e:
            print(f"Could not process {set_name} from {case_file}. Reason: {e}")

    print("\nData preparation complete.")

if __name__ == "__main__":
    args = get_args()
    preprocess(args)

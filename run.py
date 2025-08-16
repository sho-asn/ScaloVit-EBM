import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from utils.utils_data import get_full_mfp_dataloader
from model import init_wav_embedder, split_image_into_chunks
from img_transformations import WAVEmbedder

# --- Configuration ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
data_path = Path("Datasets") / "CVACaseStudy" / "MFP" / "Training.mat"

# Select a single feature to process and visualize
# You can change this index to inspect other features.
FEATURE_INDEX = 13

# The desired width of your wavelet image chunks for the DL model
CHUNK_WIDTH = 32

# --- 1. Load The Full Training Signal ---
print("Loading full training signal...")
full_train_dataloader = get_full_mfp_dataloader(
    data_path=data_path,
    sensor="T1",
    split="train",
    batch_size=1,  # Get one batch that contains the full signal
    split_ratios=(0.6, 0.2),
    shuffle=False,
    device=device
)

full_train_signal = next(iter(full_train_dataloader))[0]
print(f"Full training signal loaded with shape: {full_train_signal.shape}")

# --- 2. Initialize and Prepare the Wavelet Embedder ---
# Select the single feature we want to process
signal_feature = full_train_signal[:, :, FEATURE_INDEX:FEATURE_INDEX+1]
print(f"Selected feature slice with shape: {signal_feature.shape}")

full_signal_length = signal_feature.shape[1]

embedder = WAVEmbedder(
    device=device, 
    seq_len=full_signal_length, 
    wavelet_name='morl', 
    scales_arange=(1, 128)
)

print("Caching min/max normalization parameters for magnitude and phase...")
init_wav_embedder(embedder, signal_feature.cpu().numpy())
print("Normalization parameters cached.")

# --- 3. Transform Signal to 2-Channel CWT Image (Magnitude & Phase) ---
print("Transforming signal to wavelet image...")
wavelet_image = embedder.ts_to_img(signal_feature)
print(f"Full wavelet image shape: {wavelet_image.shape}")

# --- 4. Extract Scalogram for Plotting ---
# The first channel (index 0) is the normalized magnitude (scalogram)
scalogram = wavelet_image[:, 0, :, :].squeeze()

# --- 5. Plotting for Verification ---
print("Generating plots...")
# fig, axs = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 3]})

# # Plot 1: Original Time Series Signal
# axs[0].plot(signal_feature.squeeze().cpu().numpy(), label=f'Feature {FEATURE_INDEX}')
# axs[0].set_title(f"Original Signal for Feature {FEATURE_INDEX}")
# axs[0].set_xlabel("Time Steps")
# axs[0].set_ylabel("Amplitude")
# axs[0].legend()
# axs[0].grid(True)

# # Plot 2: Normalized Scalogram (Magnitude Channel)
# im = axs[1].imshow(scalogram.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
# axs[1].set_title(f"Normalized Wavelet Scalogram for Feature {FEATURE_INDEX}")
# axs[1].set_xlabel("Time Steps")
# axs[1].set_ylabel("Scales")
# fig.colorbar(im, ax=axs[1], label='Normalized Magnitude')

# plt.tight_layout()
# plt.show()

# --- 6. Chunking the Image and Plotting for a DL Model ---
print(f"Chunking the full wavelet image into chunks of width {CHUNK_WIDTH}...")
wavelet_image_chunks = split_image_into_chunks(wavelet_image, CHUNK_WIDTH)
print(f"Shape of wavelet image chunks: {wavelet_image_chunks.shape}")

if wavelet_image_chunks.shape[0] > 0:
    # --- 7. Plotting Scalogram for Each Chunk ---
    num_chunks_to_plot = min(5, wavelet_image_chunks.shape[0])
    print(f"Generating scalogram plots for the first {num_chunks_to_plot} chunks...")

    for i in range(num_chunks_to_plot):
        # A chunk has shape: (C, H, W_chunk).
        # For a single feature, C=2 [magnitude, phase]. We plot the magnitude (index 0).
        chunk_magnitude = wavelet_image_chunks[i, 0, :, :]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(chunk_magnitude.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f"Scalogram for Chunk {i + 1}")
        ax.set_xlabel("Time Steps (in chunk)")
        ax.set_ylabel("Scales")
        fig.colorbar(im, ax=ax, label='Normalized Magnitude')
        plt.tight_layout()
        plt.show()

    wavelet_image_dataset = TensorDataset(wavelet_image_chunks)
    wavelet_image_dataloader = DataLoader(wavelet_image_dataset, batch_size=32, shuffle=True)
    print(f"\nSuccessfully created a DataLoader with {len(wavelet_image_dataloader)} batches of chunks.")

else:
    print("No wavelet chunks were created. The signal is shorter than the chunk width.")

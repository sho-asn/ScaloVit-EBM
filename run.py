import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.io as scio

from model import init_wav_embedder, split_image_into_chunks
from img_transformations import WAVEmbedder

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_DATA_PATH = Path("Datasets") / "CVACaseStudy" / "MFP" / "Training.mat"

CHUNK_WIDTH = 64
TRAIN_SPLIT_RATIO = 0.8

# --- 1. Load T1, T2, T3 as Separate Raw Signals ---
data = scio.loadmat(TRAINING_DATA_PATH)
T1_raw = np.delete(data['T1'], -1, axis=1)
T2_raw = np.delete(data['T2'], -1, axis=1)
T3_raw = np.delete(data['T3'], -1, axis=1)
print(f"Loaded T1, T2, T3 with shapes: {T1_raw.shape}, {T2_raw.shape}, {T3_raw.shape}")

# --- 2. Split Each Signal into Train and Validation Sets ---
all_train_raw = []
all_val_raw = []

for signal_np in [T1_raw, T2_raw, T3_raw]:
    split_point = int(signal_np.shape[0] * TRAIN_SPLIT_RATIO)
    all_train_raw.append(signal_np[:split_point, :])
    all_val_raw.append(signal_np[split_point:, :])

# --- 3. Initialize Embedder and Fit on COMBINED TRAINING DATA ---
# Temporarily combine raw training signals to learn a unified normalization
combined_train_raw_np = np.concatenate(all_train_raw, axis=0)
combined_train_raw = torch.from_numpy(combined_train_raw_np).unsqueeze(0).float().to(device)
print(f"Combined raw training data to fit scaler, shape: {combined_train_raw.shape}")

# Initialize the embedder
embedder = WAVEmbedder(
    device=device, 
    seq_len=combined_train_raw.shape[1], # Seq len based on combined training set
    wavelet_name='morl', 
    scales_arange=(1, 128)
)

# Cache min/max normalization parameters using ONLY the combined training signal
print("Caching normalization parameters from all training data...")
init_wav_embedder(embedder, combined_train_raw.cpu().numpy())
print("Normalization parameters cached.")

# --- 4. Transform and Chunk Each Dataset Individually ---
print("Transforming and chunking all data sets...")

final_train_chunks = []
for train_set_np in all_train_raw:
    train_tensor = torch.from_numpy(train_set_np).unsqueeze(0).float().to(device)
    embedder.seq_len = train_tensor.shape[1]
    wavelet_image = embedder.ts_to_img(train_tensor)
    chunks = split_image_into_chunks(wavelet_image, CHUNK_WIDTH)
    final_train_chunks.append(chunks)

final_val_chunks = []
for val_set_np in all_val_raw:
    val_tensor = torch.from_numpy(val_set_np).unsqueeze(0).float().to(device)
    embedder.seq_len = val_tensor.shape[1]
    wavelet_image = embedder.ts_to_img(val_tensor)
    chunks = split_image_into_chunks(wavelet_image, CHUNK_WIDTH)
    final_val_chunks.append(chunks)

# --- 5. Concatenate All Chunks and Save ---
final_train_chunks = torch.cat(final_train_chunks, dim=0)
final_val_chunks = torch.cat(final_val_chunks, dim=0)

train_save_path = Path("preprocessed_dataset/train_chunks.pt")
torch.save(final_train_chunks, train_save_path)
print(f"Saved {final_train_chunks.shape[0]} total training chunks to {train_save_path}")

val_save_path = Path("preprocessed_dataset/val_chunks.pt")
torch.save(final_val_chunks, val_save_path)
print(f"Saved {final_val_chunks.shape[0]} total validation chunks to {val_save_path}")

# --- 6. Plot for Verification ---
if final_train_chunks.shape[0] > 0:
    print("Generating verification plot for the first training chunk...")
    first_chunk_mag = final_train_chunks[21, 0, :, :]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(first_chunk_mag.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    ax.set_title("Scalogram of First Training Chunk (Feature 0)")
    ax.set_xlabel("Time Steps (in chunk)")
    ax.set_ylabel("Scales")
    fig.colorbar(im, ax=ax, label='Normalized Magnitude')
    plt.tight_layout()
    plt.show()

print("\nData preparation complete.")
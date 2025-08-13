import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from utils.utils_data import get_mfp_dataloader, get_full_mfp_dataloader
from model import init_stft_embedder, init_wav_embedder, get_full_signal_from_dataloader, split_image_into_chunks
from img_transformations import STFTEmbedder, WAVEmbedder_ST, WAVEmbedder


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data_path = Path("Datasets")/"CVACaseStudy"/"MFP"/"Training.mat"
# chunk_size = 1024
# batch_size = 32

# train_loader = get_mfp_dataloader(
#     data_path=data_path,
#     sensor="T1",
#     split="train",
#     chunk_size=chunk_size,
#     batch_size=batch_size,
#     split_ratios=(0.6, 0.2),
#     shuffle=False)

# embedder = STFTEmbedder(device=device, seq_len=chunk_size, n_fft=63, hop_length=32)
# init_stft_embedder(embedder, train_loader)  # Normalize using entire dataset

# data = next(iter(train_loader))    
# signal = data[0].to(device)  # shape: (B, L, C)
# spectrograms = embedder.ts_to_img(signal)
# print("Spectrogram shape:", spectrograms.shape)

# the number of split = T / t (wavelet )
# Whole -> cut
# 


# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# Define data path and parameters
data_path = Path("Datasets")/"CVACaseStudy"/"MFP"/"Training.mat"
final_image_chunk_width = 32 # The desired width of your wavelet image chunks

# Load the full training data first to get global min/max for normalization
# We use a large chunk_size to effectively load the whole signal as one "chunk"
# for initial processing.
full_train_dataloader = get_full_mfp_dataloader(
    data_path=data_path,
    sensor="T1",
    split="train",
    batch_size=1, # Get one batch that ideally contains the full signal
    split_ratios=(0.6, 0.2), # (train_ratio, valid_ratio)
    shuffle=False,
    device=device
)

full_train_signal_tensor = get_full_signal_from_dataloader(full_train_dataloader)
# print(f"Full training signal loaded with shape: {full_train_signal_tensor.shape}")

# Initialize the WAVEmbedder_ST with an arbitrary seq_len; it will be updated by ts_to_img
# We set seq_len to the actual length of the full signal for embedding.
full_signal_length = full_train_signal_tensor.shape[1] 
embedder_wav = WAVEmbedder_ST(device=device, seq_len=full_signal_length, nv=7, scales='log-piecewise')


# single_feature = full_train_signal_tensor[:, :, 1]
# single_feature = full_train_signal_tensor[:, :, 1:2]  # shape: [1, 6223, 1]

# Cache min/max parameters using the entire training signal (as numpy for ssqueezepy)
print("Caching min/max parameters for WAVEmbedder_ST using the full training signal...")
signal_features = full_train_signal_tensor[:, :, 1:3]  # shape: [1, 6223, 1]
# init_wav_embedder(embedder_wav, full_train_signal_tensor.cpu().numpy())
init_wav_embedder(embedder_wav, signal_features.cpu().numpy())
print("Min/max parameters cached.")

# --- Process the entire signal ---
# Now, load the specific signal you want to transform (e.g., from validation or test set)
# Let's use the validation set for demonstration.
# full_valid_dataloader = get_full_mfp_dataloader(
#     data_path=data_path,
#     sensor="T1",
#     split="valid",
#     batch_size=1, # Get one batch that ideally contains the full signal
#     split_ratios=(0.6, 0.2), # (train_ratio, valid_ratio)
#     shuffle=False,
#     device=device
# )
# full_signal_to_transform = get_full_signal_from_dataloader(full_valid_dataloader)
# print(f"Signal to transform (validation) shape: {full_signal_to_transform.shape}")

# # Important: Update embedder's seq_len for the signal about to be transformed
# embedder_wav.seq_len = full_signal_to_transform.shape[1]

# Transform the ENTIRE signal into its wavelet image representation
# print("Transforming the entire signal into wavelet image...")
# full_wavelet_image = embedder_wav.ts_to_img(full_signal_to_transform)
# print(f"Full wavelet image shape: {full_wavelet_image.shape}")


full_wavelet_image = embedder_wav.ts_to_img(signal_features)
print(f"Full wavelet image shape: {full_wavelet_image.shape}")

# --- Separate the full wavelet image into chunks ---
print(f"Chunking the full wavelet image into chunks of width {final_image_chunk_width}...")
wavelet_image_chunks = split_image_into_chunks(full_wavelet_image, final_image_chunk_width)
print(f"Shape of wavelet image chunks: {wavelet_image_chunks.shape}")

# Now you have `wavelet_image_chunks` which can be used in a DataLoader
# Example: Create a DataLoader for the wavelet image chunks
wavelet_image_dataset = TensorDataset(wavelet_image_chunks)
wavelet_image_dataloader = DataLoader(wavelet_image_dataset, batch_size=32, shuffle=True)

print(f"Number of batches in chunked wavelet image dataloader: {len(wavelet_image_dataloader)}")
first_batch_of_chunks = next(iter(wavelet_image_dataloader))
print(f"Shape of first batch of wavelet image chunks: {first_batch_of_chunks[0].shape}")


if wavelet_image_chunks.shape[0] > 0:
    # Take the first chunk for visualization
    sample_wavelet_chunk = wavelet_image_chunks[0].unsqueeze(0) # Add batch dimension back for img_to_ts
    print(f"Sample chunk shape for reconstruction: {sample_wavelet_chunk.shape}")

    # Pick the first sample in the batch (batch dim = 0)
    wavelet_img = full_wavelet_image[0]  # shape: (C, scales, time)

    # Pick the real part of feature 0 → channel index 0
    real_part = wavelet_img[0].cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(real_part, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Wavelet Transform - Real Part (Feature 0)')
    plt.show()

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from utils.utils_data import get_mfp_dataloader
from model import init_stft_embedder
from img_transformations import STFTEmbedder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = Path("Datasets")/"CVACaseStudy"/"MFP"/"Training.mat"
chunk_size = 1024
batch_size = 32

train_loader = get_mfp_dataloader(
    data_path=data_path,
    sensor="T1",
    split="train",
    chunk_size=chunk_size,
    batch_size=batch_size,
    split_ratios=(0.6, 0.2),
    shuffle=False)

embedder = STFTEmbedder(device=device, seq_len=chunk_size, n_fft=63, hop_length=32)
init_stft_embedder(embedder, train_loader)  # Normalize using entire dataset

data = next(iter(train_loader))    
signal = data[0].to(device)  # shape: (B, L, C)
spectrograms = embedder.ts_to_img(signal)
print("Spectrogram shape:", spectrograms.shape)


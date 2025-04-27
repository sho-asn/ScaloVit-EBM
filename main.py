import torch
import matplotlib.pyplot as plt
from pathlib import Path
from utils.utils_data import load_mat_data, split_data, split_into_chunks
from utils.utils_visualization import save_feature_plots_to_pdf
from img_transformations import STFTEmbedder


data_path = Path("Datasets")/"CVACaseStudy"/"MFP"/"Training.mat"
data_t1, _, _ = load_mat_data(data_path, ["T1", "T2", "T3"])
train_t1, valid_t1, test_t1 = split_data(data_t1, train_ratio=0.6, valid_ratio=0.2)
train_t1_tensor = torch.tensor(train_t1, dtype=torch.float32).unsqueeze(0)  # shape: (Batch(1), Length, Feature)
print("Input shape:", train_t1_tensor.shape)

chunk_size = 128
train_t1_chunks = split_into_chunks(train_t1_tensor, chunk_size) 
print("Splitted chunk shape: ", train_t1_chunks.shape) # shape: (Batch, Length, Feature)

# Initialize embedder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 64
n_fft = 63
hop_length = 23
embedder = STFTEmbedder(device, seq_len, n_fft, hop_length)
real, imag = embedder.stft_transform(train_t1_tensor)

# Normalize + convert to spectrogram
embedder.cache_min_max_params(train_t1_tensor)  # init normalization
train_t1_chunks = train_t1_chunks.to(device)  
spectrograms = embedder.ts_to_img(train_t1_chunks)  # shape: (NumChunks, 2*C, Freq, Time)
print("spectrogram shape: ", spectrograms.shape)


save_feature_plots_to_pdf(
    signal=train_t1_chunks,
    spectrogram=spectrograms,
    feature_idx=1,
    pdf_path=Path("results")/"plots"/"spectrograms_full_f1_trainT1.pdf",
    vmin=-1,
    vmax=1
)

# # Visualize one channel of the spectrogram 
# real_channel = spectrograms[0, 0].cpu().detach().numpy()
# plt.imshow(real_channel, aspect='auto', origin='lower')
# plt.title("STFT Spectrogram (Real Part, Feature 1)")
# plt.colorbar()
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()
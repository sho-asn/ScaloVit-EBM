import torch
import matplotlib.pyplot as plt
from pathlib import Path
from utils.utils_data import load_mat_data, split_data, split_into_chunks, inject_anomalies, get_mfp_dataloader
from utils.utils_visualization import save_feature_plots_to_pdf
from img_transformations import STFTEmbedder


# data_path = Path("Datasets")/"CVACaseStudy"/"MFP"/"Training.mat"
# data_t1, _, _ = load_mat_data(data_path, ["T1", "T2", "T3"])
# train_t1, valid_t1, test_t1 = split_data(data_t1, train_ratio=0.6, valid_ratio=0.2)
# train_t1_tensor = torch.tensor(train_t1, dtype=torch.float32).unsqueeze(0)  # shape: (Batch(1), Length, Feature)
# print("Input shape:", train_t1_tensor.shape)

# chunk_size = 1024
# train_t1_chunks = split_into_chunks(train_t1_tensor, chunk_size) 
# print("Splitted chunk shape: ", train_t1_chunks.shape) # shape: (Batch, Length, Feature)

# Add anomaly
# train_t1_chunks = inject_anomalies(train_t1_chunks, fault_type="bias", batch_idx=4, selected_features=[1]) # bias
# train_t1_chunks = inject_anomalies(train_t1_chunks, fault_type="drift", batch_idx=4, selected_features=[1]) # drift
# train_t1_chunks = inject_anomalies(train_t1_chunks, fault_type="erratic", batch_idx=4, selected_features=[1]) # erratic
# train_t1_chunks = inject_anomalies(train_t1_chunks, fault_type="spike", batch_idx=4, selected_features=[1]) # spike
# train_t1_chunks = inject_anomalies(train_t1_chunks, fault_type="stuck", batch_idx=4, selected_features=[1]) # stuck


# # Initialize embedder
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seq_len = chunk_size
# n_fft = 63 
# hop_length = 32
# embedder = STFTEmbedder(device, seq_len, n_fft, hop_length)
# real, imag = embedder.stft_transform(train_t1_tensor)

# # Normalize + convert to spectrogram
# embedder.cache_min_max_params(train_t1_tensor)  # init normalization
# train_t1_chunks = train_t1_chunks.to(device)  
# spectrograms = embedder.ts_to_img(train_t1_chunks)  # shape: (NumChunks, 2*C, Freq, Time)
# print("spectrogram shape: ", spectrograms.shape)


# save_feature_plots_to_pdf(
#     signal=train_t1_chunks,
#     spectrogram=spectrograms,
#     feature_idx=1,
#     pdf_path=Path("results")/"plots"/"spectrograms_full_f1_trainT1_anomaly.pdf",
#     vmin=-1,
#     vmax=1
# )

# init the min and max values for the STFTEmbedder, this function must be called before the training loop starts
def init_stft_embedder(embedder, train_loader):
    """
    Initializes min/max values for normalization across the whole dataset.
    Args:
        embedder (STFTEmbedder): the embedder object.
        train_loader (DataLoader): training data loader.
    """
    data = []
    for data_batch in train_loader:
        data.append(data_batch[0])  # Extract input tensor from batch tuple
    all_data = torch.cat(data, dim=0)  # Concatenate along batch dimension
    print(all_data.shape)
    embedder.cache_min_max_params(torch.cat(data, dim=0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = Path("Datasets")/"CVACaseStudy"/"MFP"/"Training.mat"
chunk_size = 1024
batch_size = 32

# Load data
train_loader = get_mfp_dataloader(
    data_path=data_path,
    sensor="T1",
    split="train",
    chunk_size=chunk_size,
    batch_size=batch_size,
    split_ratios=(0.6, 0.2),
    shuffle=False)

# Initialize embedder
embedder = STFTEmbedder(device=device, seq_len=chunk_size, n_fft=63, hop_length=32)
init_stft_embedder(embedder, train_loader)  # Normalize using entire dataset

for batch in train_loader:
    signal = batch[0].to(device)  # shape: (B, L, C)
    spectrograms = embedder.ts_to_img(signal)
    print("Spectrogram shape:", spectrograms.shape)
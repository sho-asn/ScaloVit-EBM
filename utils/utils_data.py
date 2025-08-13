import torch
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader



def load_mat_data(file_path: Path, variables: List[str]) -> List[np.ndarray]:
    """
    Load soft sensor data which consists of three types of data (T1, T2, T3)
    """
    data = scio.loadmat(file_path)
    return [data[var] for var in variables]

def split_data(data: np.ndarray, train_ratio: float=0.6, valid_ratio: float=0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_size = data.shape[0]
    train_end = int(total_size * train_ratio)
    valid_end = int(total_size * (train_ratio + valid_ratio))

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]

    return train_data, valid_data, test_data

def MinMaxScaler(data, return_scalers=False):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    min = np.min(data, 0)
    max = np.max(data, 0)
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min, max
    return norm_data


def MinMaxArgs(data, min, max):
    """
    Args:
        data: given data
        min: given min value
        max: given max value

    Returns:
        min-max scaled data by given min and max
    """
    numerator = data - min
    denominator = max - min
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def plot_signal(data: np.ndarray, save_path: Path|None = None):
    num_features = data.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(14, 2 * num_features))
    
    if num_features == 1:
        axes = [axes]  # Ensure axes is iterable

    for i in range(num_features):
        axes[i].plot(data[:, i], label=f'Feature {i+1}')
        axes[i].set_ylabel(f'F{i+1}')
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="pdf")
    plt.show()

def split_into_chunks(data: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch_size, seq_len, features = data.shape
    n_chunks = seq_len // chunk_size  # Number of full chunks

    # Truncate the signal to a multiple of chunk_size
    data = data[:, :n_chunks * chunk_size, :]

    # Reshape
    chunks = data.reshape(batch_size * n_chunks, chunk_size, features)

    return chunks


def get_mfp_dataloader(
        data_path, 
        sensor="T1", 
        split="train", 
        chunk_size=1024,
        batch_size=32, 
        split_ratios=(0.6, 0.2), # (train_ratio, valid_ratio)
        shuffle=False, 
        num_workers=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

    T1, T2, T3 = load_mat_data(data_path, ["T1", "T2", "T3"])
    data_dict = {"T1": T1, "T2": T2, "T3": T3}
    train_data, valid_data, test_data = split_data(data_dict[sensor], *split_ratios)

    if split == "train":
        data = train_data
    elif split == "valid":
        data = valid_data
    elif split == "test":
        data = test_data

    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    chunks = split_into_chunks(data_tensor, chunk_size)
    chunks = chunks.to(device)
    dataloader = DataLoader(TensorDataset(chunks), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def get_full_mfp_dataloader(
        data_path, 
        sensor="T1", 
        split="train", 
        batch_size=32, 
        split_ratios=(0.6, 0.2), # (train_ratio, valid_ratio)
        shuffle=False, 
        num_workers=0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):

    T1, T2, T3 = load_mat_data(data_path, ["T1", "T2", "T3"])
    data_dict = {"T1": T1, "T2": T2, "T3": T3}
    train_data, valid_data, test_data = split_data(data_dict[sensor], *split_ratios)

    if split == "train":
        data = train_data
    elif split == "valid":
        data = valid_data
    elif split == "test":
        data = test_data

    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    data_tensor = data_tensor.to(device)
    dataloader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def apply_bias(signal: torch.Tensor, magnitude: float) -> torch.Tensor:
    return signal + magnitude

def apply_drift(signal: torch.Tensor, magnitude: float) -> torch.Tensor:
    drift = torch.linspace(0, magnitude, steps=signal.shape[0], device=signal.device)
    if signal.ndim == 2:
        drift = drift.unsqueeze(-1)
    return signal + drift

def apply_erratic(signal: torch.Tensor, noise_level: float) -> torch.Tensor:
    noise = torch.randn_like(signal) * noise_level
    return signal + noise

def apply_spike(signal: torch.Tensor, magnitude: float, num_spikes: int = 5) -> torch.Tensor:
    signal = signal.clone()
    time_steps = signal.shape[0]
    if signal.ndim == 2:
        feature_dim = signal.shape[1]
        for _ in range(num_spikes):
            t_idx = random.randint(0, time_steps - 1)
            f_idx = random.randint(0, feature_dim - 1)
            signal[t_idx, f_idx] += magnitude * (2 * torch.rand(1).item() - 1)
    else:
        for _ in range(num_spikes):
            t_idx = random.randint(0, time_steps - 1)
            signal[t_idx] += magnitude * (2 * torch.rand(1).item() - 1)
    return signal

def apply_stuck(signal: torch.Tensor) -> torch.Tensor:
    stuck_value = signal[random.randint(0, signal.shape[0] - 1)]
    if signal.ndim == 2:
        stuck_value = stuck_value.unsqueeze(0)
        return stuck_value.repeat(signal.shape[0], 1)
    else:
        return stuck_value.repeat(signal.shape[0])

def inject_anomalies(
        signal: torch.Tensor,
        fault_type: str,
        batch_idx: Optional[int] = None,
        magnitude: float = 0.10,
        noise_level: float = 0.01,
        selected_features: Optional[List[int]] = None,
        stuck_probability: float = 1.0,
        spike_num: int = 5) -> torch.Tensor:
    """
    Inject anomalies into a selected batch of the input signal.

    Args:
        signal: Tensor of shape (B, L, F)
        fault_type: One of ['bias', 'drift', 'erratic', 'spike', 'stuck']
        batch_idx: Which batch index to apply anomaly to. (Required)
        magnitude: Size of bias, drift, or spike.
        noise_level: Standard deviation for erratic noise.
        selected_features: List of feature indices to apply fault. Default: all features.
        stuck_probability: Probability for a feature to be stuck if 'stuck' fault.
        spike_num: Number of spikes if 'spike' fault.

    Returns:
        Tensor with injected anomaly.
    """
    if batch_idx is None:
        raise ValueError("batch_idx must be specified when using batch input.")

    signal = signal.clone()

    B, L, F = signal.shape

    if not (0 <= batch_idx < B):
        raise IndexError(f"batch_idx {batch_idx} is out of range for batch size {B}")

    for f in range(F):
        if (selected_features is None) or (f in selected_features):
            if fault_type == "bias":
                signal[batch_idx, :, f] = apply_bias(signal[batch_idx, :, f], magnitude)
            elif fault_type == "drift":
                signal[batch_idx, :, f] = apply_drift(signal[batch_idx, :, f], magnitude)
            elif fault_type == "erratic":
                signal[batch_idx, :, f] = apply_erratic(signal[batch_idx, :, f], noise_level)
            elif fault_type == "spike":
                signal[batch_idx, :, f] = apply_spike(signal[batch_idx, :, f], magnitude, spike_num)
            elif fault_type == "stuck":
                if random.random() < stuck_probability:
                    signal[batch_idx, :, f] = apply_stuck(signal[batch_idx, :, f])
            else:
                raise ValueError(f"Unsupported fault type: {fault_type}")

    return signal


# if __name__ == "__main__":
#     data_dir = Path("..")/"Datasets"/"CVACaseStudy"/"MFP"/"Training.mat"
#     data_t1, data_t2, data_t3 = load_mat_data(data_dir, ["T1", "T2", "T3"])
#     train_t1, valid_t1, test_t1 = split_data(data=data_t1, train_ratio=0.6, valid_ratio=0.2)
#     train_t2, valid_t2, test_t2 = split_data(data=data_t2, train_ratio=0.6, valid_ratio=0.2)
#     train_t3, valid_t3, test_t3 = split_data(data=data_t3, train_ratio=0.6, valid_ratio=0.2)

#     # Normalization
#     X_train_norm = MinMaxScaler(train_t1)
#     X_valid_norm = MinMaxScaler(valid_t1)
#     X_test_norm = MinMaxScaler(test_t1)

#     plot_signal(X_valid_norm, 
#                 save_path=Path("..")/"results"/"plots"/"valid_norm_signal_plot.pdf")
    
    # train_loader, valid_loader, test_loader = get_mfp_dataloader(
    #     mat_file_path=Path("..")/"Datasets"/"CVACaseStudy"/"MFP"/"Training.mat",
    #     signal_key="T1",
    #     chunk_size=1024,
    #     batch_size=32,
    #     split_ratios=(0.6, 0.2)
    # )
    # for batch in train_loader:
    #     print(batch[0].shape)

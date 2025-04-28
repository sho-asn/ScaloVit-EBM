import torch
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt


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

def apply_bias(signal: torch.Tensor, magnitude: float) -> torch.Tensor:
    return signal + magnitude

def apply_drift(signal: torch.Tensor, magnitude: float) -> torch.Tensor:
    drift = torch.linspace(0, magnitude, steps=signal.shape[0], device=signal.device)
    return signal + drift.unsqueeze(-1)

def apply_erratic(signal: torch.Tensor, noise_level: float) -> torch.Tensor:
    noise = torch.randn_like(signal) * noise_level
    return signal + noise

def apply_spike(signal: torch.Tensor, magnitude: float, num_spikes: int = 5) -> torch.Tensor:
    signal = signal.clone()
    time_steps = signal.shape[0]
    feature_dim = signal.shape[1]
    for _ in range(num_spikes):
        t_idx = random.randint(0, time_steps - 1)
        f_idx = random.randint(0, feature_dim - 1)
        signal[t_idx, f_idx] += magnitude * (2 * torch.rand(1).item() - 1)  # Random + or -
    return signal

def apply_stuck(signal: torch.Tensor) -> torch.Tensor:
    stuck_value = signal[random.randint(0, signal.shape[0] - 1)]
    return stuck_value.repeat(signal.shape[0], 1)


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
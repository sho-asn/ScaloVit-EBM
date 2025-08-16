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
    min = np.min(data, 1)
    max = np.max(data, 1)
    numerator = data - np.min(data, 1)
    denominator = np.max(data, 1) - np.min(data, 1)
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
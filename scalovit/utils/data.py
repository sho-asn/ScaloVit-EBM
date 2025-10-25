"""Data loading and preprocessing utilities used across the project."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_mat_data(file_path: Path, variables: List[str]) -> List[np.ndarray]:
    """Load specified variables from a MATLAB ``.mat`` file."""

    data = scio.loadmat(file_path)
    return [data[var] for var in variables]


def split_data(
    data: np.ndarray,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an array into train/validation/test partitions."""

    total_size = data.shape[0]
    train_end = int(total_size * train_ratio)
    valid_end = int(total_size * (train_ratio + valid_ratio))
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    return train_data, valid_data, test_data


def MinMaxScaler(data: np.ndarray, return_scalers: bool = False):
    """Classic min-max scaling with optional scaler return."""

    min_vals = np.min(data, 1)
    max_vals = np.max(data, 1)
    numerator = data - min_vals
    denominator = max_vals - min_vals
    norm_data = numerator / (denominator + 1e-7)
    if return_scalers:
        return norm_data, min_vals, max_vals
    return norm_data


def MinMaxArgs(data: np.ndarray | torch.Tensor, min_vals, max_vals):
    """Apply min-max scaling using precomputed min/max tensors."""

    numerator = data - min_vals
    denominator = max_vals - min_vals
    return numerator / (denominator + 1e-7)


def plot_signal(data: np.ndarray, save_path: Path | None = None) -> None:
    """Plot multivariate time-series with one subplot per feature."""

    num_features = data.shape[1]
    fig, axes = plt.subplots(num_features, 1, figsize=(14, 2 * num_features))
    if num_features == 1:
        axes = [axes]

    for idx in range(num_features):
        axes[idx].plot(data[:, idx], label=f"Feature {idx + 1}")
        axes[idx].set_ylabel(f"F{idx + 1}")
        axes[idx].legend(loc="upper right")
        axes[idx].grid(True)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format="pdf")
    plt.show()


def split_into_chunks(data: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Split signals into fixed-width chunks along the temporal dimension."""

    batch_size, seq_len, features = data.shape
    n_chunks = seq_len // chunk_size
    data = data[:, : n_chunks * chunk_size, :]
    chunks = data.reshape(batch_size * n_chunks, chunk_size, features)
    return chunks


def get_mfp_dataloader(
    data_path: Path,
    sensor: str = "T1",
    split: str = "train",
    chunk_size: int = 1024,
    batch_size: int = 32,
    split_ratios: Tuple[float, float] = (0.6, 0.2),
    shuffle: bool = False,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> DataLoader:
    """Return dataloaders for the CVA case study dataset with chunked windows."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t1, t2, t3 = load_mat_data(data_path, ["T1", "T2", "T3"])
    data_dict = {"T1": t1, "T2": t2, "T3": t3}
    train_data, valid_data, test_data = split_data(data_dict[sensor], *split_ratios)

    if split == "train":
        selected = train_data
    elif split == "valid":
        selected = valid_data
    else:
        selected = test_data

    data_tensor = torch.tensor(selected, dtype=torch.float32).unsqueeze(0)
    chunks = split_into_chunks(data_tensor, chunk_size).to(device)
    return DataLoader(TensorDataset(chunks), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_full_mfp_dataloader(
    data_path: Path,
    sensor: str = "T1",
    split: str = "train",
    batch_size: int = 32,
    split_ratios: Tuple[float, float] = (0.6, 0.2),
    shuffle: bool = False,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> DataLoader:
    """Return dataloaders that yield the full signal instead of chunks."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t1, t2, t3 = load_mat_data(data_path, ["T1", "T2", "T3"])
    data_dict = {"T1": t1, "T2": t2, "T3": t3}
    train_data, valid_data, test_data = split_data(data_dict[sensor], *split_ratios)

    if split == "train":
        selected = train_data
    elif split == "valid":
        selected = valid_data
    else:
        selected = test_data

    data_tensor = torch.tensor(selected, dtype=torch.float32).unsqueeze(0).to(device)
    return DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

from pathlib import Path
from typing import List, Tuple, Dict

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

def get_norm_param(x: np.ndarray) -> Dict[str, np.ndarray]:
    keys = ['x_min','x_max','x_mean','x_std']
    norm_param: Dict = {}
    for key in keys: 
        norm_param[key] = []
    
    norm_param['x_min']  = np.min(x, axis=0)
    norm_param['x_max']  = np.max(x, axis=0)
    norm_param['x_mean'] = np.mean(x, axis=0)
    norm_param['x_std']  = np.std(x, axis=0)

    return norm_param

def normalize(x: np.ndarray, norm_param: Dict[str, np.ndarray], method: str) -> np.ndarray: 
    if method == 'minmax':
        x_norm = (x - norm_param['x_min']) / (norm_param['x_max'] - norm_param['x_min'])
    elif method == 'standardize':
        x_norm = (x - norm_param['x_mean']) / norm_param['x_std']
    else:
        raise TypeError("Normalization method not known")
    
    return x_norm

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


if __name__ == "__main__":
    data_dir = Path("..")/"Datasets"/"CVACaseStudy"/"MFP"/"Training.mat"
    data_t1, data_t2, data_t3 = load_mat_data(data_dir, ["T1", "T2", "T3"])
    train_t1, valid_t1, test_t1 = split_data(data=data_t1, train_ratio=0.6, valid_ratio=0.2)
    train_t2, valid_t2, test_t2 = split_data(data=data_t2, train_ratio=0.6, valid_ratio=0.2)
    train_t3, valid_t3, test_t3 = split_data(data=data_t3, train_ratio=0.6, valid_ratio=0.2)

    # Normalization
    norm_param_train = get_norm_param(train_t1)
    norm_method = "minmax"
    X_train_norm = normalize(train_t1, norm_param_train, norm_method)
    X_valid_norm = normalize(valid_t1, norm_param_train, norm_method)
    X_test_norm = normalize(test_t1, norm_param_train, norm_method)

    plot_signal(X_valid_norm, 
                save_path=Path("..")/"results"/"plots"/"train_norm_signal_plot.pdf")
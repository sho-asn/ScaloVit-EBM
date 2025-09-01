import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Optional, Union, List, Tuple
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def plot_energy_with_anomalies(
    energy_scores: Union[torch.Tensor, np.ndarray],
    threshold: float,
    save_path: Union[str, Path],
    title: str = "Energy Scores with Anomaly Threshold",
    ground_truth_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
):
    """
    Plots the energy scores and highlights anomalous regions based on ground truth or predictions.

    Args:
        energy_scores (Union[torch.Tensor, np.ndarray]): A 1D tensor or array of energy scores.
        threshold (float): The anomaly threshold.
        save_path (Union[str, Path]): The path to save the plot.
        title (str, optional): The title of the plot. Defaults to 'Energy Scores with Anomaly Threshold'.
        ground_truth_labels (Optional[Union[torch.Tensor, np.ndarray]], optional): 
            A 1D tensor or array of ground truth labels (1 for anomaly, 0 for normal). 
            If provided, these will be used to highlight anomalous regions. 
            Defaults to None, in which case predicted anomalies are highlighted.
    """
    if isinstance(energy_scores, torch.Tensor):
        energy_scores = energy_scores.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(energy_scores, label="Energy Score")
    plt.axhline(y=threshold, color="r", linestyle="--", label="Anomaly Threshold")

    highlight_label = ""
    if ground_truth_labels is not None:
        if isinstance(ground_truth_labels, torch.Tensor):
            ground_truth_labels = ground_truth_labels.cpu().numpy()
        anomalous_indices = np.where(ground_truth_labels == 1)[0]
        highlight_label = "Ground Truth Anomaly"
    else:
        anomalous_indices = np.where(energy_scores > threshold)[0]
        highlight_label = "Predicted Anomaly"

    if len(anomalous_indices) > 0:
        # Find contiguous regions of anomalies
        regions = []
        start = anomalous_indices[0]
        for i in range(1, len(anomalous_indices)):
            if anomalous_indices[i] != anomalous_indices[i - 1] + 1:
                regions.append((start, anomalous_indices[i - 1]))
                start = anomalous_indices[i]
        regions.append((start, anomalous_indices[-1]))

        # Add a single legend entry for all anomalous regions
        plt.axvspan(
            regions[0][0],
            regions[0][1] + 1,
            color="red",
            alpha=0.3,
            label=highlight_label,
        )
        for start, end in regions[1:]:
            plt.axvspan(start, end + 1, color="red", alpha=0.3)

    plt.title(title)
    plt.xlabel("Chunk Index")
    plt.ylabel("Energy Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_stft_images(
    stft_chunks: torch.Tensor,
    save_path: Union[str, Path],
    title: str = "STFT Preprocessed Images",
    num_images_to_plot: int = 5,
    num_features_to_plot: int = 3,
):
    """
    Plots and saves a grid of STFT images from a batch of chunks.

    Args:
        stft_chunks (torch.Tensor): A 4D tensor of STFT chunks.
                                    Shape: (N, C, H, W).
        save_path (Union[str, Path]): The path to save the plot.
        title (str, optional): The title of the plot.
        num_images_to_plot (int, optional): The number of images (chunks) to plot.
        num_features_to_plot (int, optional): The number of features to visualize.
    """
    if isinstance(stft_chunks, torch.Tensor):
        stft_chunks = stft_chunks.cpu().numpy()

    num_images_to_plot = min(num_images_to_plot, stft_chunks.shape[0])
    num_features_to_plot = min(num_features_to_plot, stft_chunks.shape[1] // 2)

    if num_images_to_plot == 0 or num_features_to_plot == 0:
        print("Not enough images or features to plot.")
        return

    # We will plot magnitude and phase for each feature
    fig, axes = plt.subplots(
        num_features_to_plot * 2,
        num_images_to_plot,
        figsize=(num_images_to_plot * 3, num_features_to_plot * 6),
    )
    if num_images_to_plot == 1 and num_features_to_plot == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif num_images_to_plot == 1:
        axes = axes.reshape(num_features_to_plot * 2, 1)
    elif num_features_to_plot == 1:
        axes = axes.reshape(2, num_images_to_plot)


    fig.suptitle(title, fontsize=16)

    for i in range(num_images_to_plot):
        for j in range(num_features_to_plot):
            mag_channel_idx = j * 2
            phase_channel_idx = j * 2 + 1

            # Plot Magnitude
            ax_mag = axes[j * 2, i]
            im_mag = ax_mag.imshow(stft_chunks[i, mag_channel_idx, :, :], cmap="viridis", aspect="auto", origin='lower')
            ax_mag.set_title(f"Chunk {i+1} / Feat {j+1} Mag")
            ax_mag.set_xlabel("Time")
            ax_mag.set_ylabel("Frequency")
            fig.colorbar(im_mag, ax=ax_mag)

            # Plot Phase
            ax_phase = axes[j * 2 + 1, i]
            im_phase = ax_phase.imshow(stft_chunks[i, phase_channel_idx, :, :], cmap="viridis", aspect="auto", origin='lower')
            ax_phase.set_title(f"Chunk {i+1} / Feat {j+1} Phase")
            ax_phase.set_xlabel("Time")
            ax_phase.set_ylabel("Frequency")
            fig.colorbar(im_phase, ax=ax_phase)


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved STFT plot to {save_path}")


"""Visualization utilities for anomaly detection experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_energy_with_anomalies(
    energy_scores: Union[torch.Tensor, np.ndarray],
    threshold: float,
    save_path: Union[str, Path],
    title: str = "Energy Scores with Anomaly Threshold",
    ground_truth_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> None:
    """Plot energy scores together with ground-truth or predicted anomalies."""

    if isinstance(energy_scores, torch.Tensor):
        energy_scores = energy_scores.cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(energy_scores, label="Energy Score")
    plt.axhline(y=threshold, color="r", linestyle="--", label="Anomaly Threshold")

    if ground_truth_labels is not None:
        if isinstance(ground_truth_labels, torch.Tensor):
            ground_truth_labels = ground_truth_labels.cpu().numpy()
        anomalous_indices = np.where(ground_truth_labels == 1)[0]
        highlight_label = "Ground Truth Anomaly"
    else:
        anomalous_indices = np.where(energy_scores > threshold)[0]
        highlight_label = "Predicted Anomaly"

    if len(anomalous_indices) > 0:
        regions = []
        start = anomalous_indices[0]
        for idx in range(1, len(anomalous_indices)):
            if anomalous_indices[idx] != anomalous_indices[idx - 1] + 1:
                regions.append((start, anomalous_indices[idx - 1]))
                start = anomalous_indices[idx]
        regions.append((start, anomalous_indices[-1]))

        plt.axvspan(regions[0][0], regions[0][1] + 1, color="red", alpha=0.3, label=highlight_label)
        for start, end in regions[1:]:
            plt.axvspan(start, end + 1, color="red", alpha=0.3)

    plt.title(title, fontsize=16)
    plt.xlabel("Chunk Index", fontsize=14)
    plt.ylabel("Energy Score", fontsize=14)
    plt.legend(fontsize=12)
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
) -> None:
    """Plot a selection of STFT chunks for quick inspection."""

    if isinstance(stft_chunks, torch.Tensor):
        stft_chunks = stft_chunks.cpu().numpy()

    num_images_to_plot = min(num_images_to_plot, stft_chunks.shape[0])
    num_features_to_plot = min(num_features_to_plot, stft_chunks.shape[1] // 2)

    if num_images_to_plot == 0 or num_features_to_plot == 0:
        print("Not enough images or features to plot.")
        return

    fig, axes = plt.subplots(num_features_to_plot * 2, num_images_to_plot, figsize=(num_images_to_plot * 3, num_features_to_plot * 6))
    if num_images_to_plot == 1 and num_features_to_plot == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif num_images_to_plot == 1:
        axes = axes.reshape(num_features_to_plot * 2, 1)
    elif num_features_to_plot == 1:
        axes = axes.reshape(2, num_images_to_plot)

    fig.suptitle(title, fontsize=16)

    for img_idx in range(num_images_to_plot):
        for feat_idx in range(num_features_to_plot):
            mag_channel_idx = feat_idx * 2
            phase_channel_idx = feat_idx * 2 + 1

            ax_mag = axes[feat_idx * 2, img_idx]
            im_mag = ax_mag.imshow(stft_chunks[img_idx, mag_channel_idx, :, :], cmap="viridis", aspect="auto", origin="lower")
            ax_mag.set_title(f"Chunk {img_idx + 1} / Feat {feat_idx + 1} Mag")
            ax_mag.set_xlabel("Time")
            ax_mag.set_ylabel("Frequency")
            fig.colorbar(im_mag, ax=ax_mag)

            ax_phase = axes[feat_idx * 2 + 1, img_idx]
            im_phase = ax_phase.imshow(stft_chunks[img_idx, phase_channel_idx, :, :], cmap="viridis", aspect="auto", origin="lower")
            ax_phase.set_title(f"Chunk {img_idx + 1} / Feat {feat_idx + 1} Phase")
            ax_phase.set_xlabel("Time")
            ax_phase.set_ylabel("Frequency")
            fig.colorbar(im_phase, ax=ax_phase)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved STFT plot to {save_path}")


def plot_scalogram_from_file(
    file_path: str,
    save_dir: str,
    chunk_index: int = 0,
    feature_index: int = 0,
) -> None:
    """Load a transformed chunk from disk and visualise the magnitude scalogram."""

    try:
        data = torch.load(file_path, map_location="cpu")
        if isinstance(data, dict):
            chunks = data["chunks"]
        else:
            chunks = data

        if not isinstance(chunks, torch.Tensor) or chunks.dim() != 4:
            print("Error: Loaded data is not a valid 4D tensor of chunks.")
            return
        if chunk_index >= len(chunks):
            print(f"Error: chunk_index {chunk_index} is out of bounds for {len(chunks)} chunks.")
            return

        the_chunk = chunks[chunk_index]
        magnitude_channel_index = 2 * feature_index
        if magnitude_channel_index >= the_chunk.shape[0]:
            print(f"Error: feature_index {feature_index} is out of bounds for the number of features.")
            return

        scalogram = the_chunk[magnitude_channel_index].numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(scalogram, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xlabel("Time Step in Chunk")
        ax.set_ylabel("Wavelet Scale")
        title = f"Scalogram (Magnitude)\nFile: {Path(file_path).name}, Chunk: {chunk_index}, Feature: {feature_index}"
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Normalized Magnitude")

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / f"scalogram_{Path(file_path).stem}_c{chunk_index}_f{feature_index}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Scalogram plot saved to {save_path}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
    except Exception as exc:  # pragma: no cover - debugging helper
        print(f"An error occurred: {exc}")

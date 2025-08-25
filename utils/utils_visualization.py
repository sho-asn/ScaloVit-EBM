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


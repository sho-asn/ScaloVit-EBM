import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import scipy.io as scio
import numpy as np

# Definition of fault intervals from plot_normal_vs_faulty.py
fault_intervals = {
    "FaultyCase1": {
        "Set1_1": [(1566, 5181)],
        "Set1_2": [(657, 3777)],
        "Set1_3": [(691, 3691)]
    },
    "FaultyCase2": {
        "Set2_1": [(2244, 6616)],
        "Set2_2": [(476, 2656)],
        "Set2_3": [(331, 2467)]
    },
    "FaultyCase3": {
        "Set3_1": [(1136, 8352)],
        "Set3_2": [(333, 5871)],
        "Set3_3": [(596, 9566)]
    },
    "FaultyCase4": {
        "Set4_1": [(953, 6294)],
        "Set4_2": [(851, 3851)],
        "Set4_3": [(241, 3241)]
    },
    "FaultyCase5": {
        "Set5_1": [(686, 1172), (1772, 2253)],
        "Set5_2": [(1633, 2955), (7031, 7553), (8057, 10608)]
    }
}

def load_normal_data(file_path: Path, normal_set_name: str):
    """Loads a specific normal dataset from the Training.mat file."""
    try:
        data = scio.loadmat(file_path)
        return data[normal_set_name]
    except FileNotFoundError:
        print(f"Error: Training file not found at {file_path}")
        return None
    except KeyError:
        print(f"Error: Normal set '{normal_set_name}' not found in the training file.")
        return None

def load_faulty_data(file_path: Path, faulty_set_name: str):
    """Loads a specific faulty dataset from a .mat file."""
    try:
        data = scio.loadmat(file_path)
        return np.delete(data[faulty_set_name], -1, axis=1)
    except FileNotFoundError:
        print(f"Error: Faulty case file not found at {file_path}")
        return None
    except KeyError:
        print(f"Error: Faulty set '{faulty_set_name}' not found in {file_path.name}.")
        return None

def plot_combined(normal_data, faulty_data, detection_df, normal_label, faulty_case_name, faulty_set_label, feature_idx, output_filename):
    """Generates a combined plot of normal data, faulty data, energy score, and distribution."""
    with plt.rc_context({'font.size': 14,
                         'axes.labelsize': 16,
                         'axes.titlesize': 18,
                         'xtick.labelsize': 14,
                         'ytick.labelsize': 14,
                         'figure.titlesize': 20}):
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
        fig.suptitle(f'Comprehensive Analysis: {normal_label} vs {faulty_case_name} {faulty_set_label} - Feature {feature_idx + 1}', y=0.98)

        # --- Subplot 1: Normal Data ---
        ax1 = axes[0]
        ax1.plot(normal_data[:, feature_idx], color='black', linewidth=1.5)
        ax1.set_title(f'Normal Signal: {normal_label}')
        ax1.set_ylabel('Value')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- Subplot 2: Faulty Data ---
        ax2 = axes[1]
        ax2.plot(faulty_data[:, feature_idx], color='steelblue', linewidth=1.5)
        ax2.set_title(f'Faulty Signal: {faulty_case_name} {faulty_set_label}')
        ax2.set_ylabel('Value')
        ax2.grid(True, linestyle='--', alpha=0.5)
        intervals = fault_intervals.get(faulty_case_name, {}).get(faulty_set_label, [])
        for start, end in intervals:
            ax2.axvspan(start, end, color='red', alpha=0.3)
            ax2.axvline(start, color='red', linestyle='--', linewidth=1)
            ax2.axvline(end, color='red', linestyle='--', linewidth=1)

        # --- Subplot 3: Energy Score Over Time ---
        ax3 = axes[2]
        ax3.plot(detection_df.index, detection_df['score'], label='Energy Score', color='steelblue', linewidth=1.5)
        ymin, ymax = ax3.get_ylim()
        ax3.fill_between(detection_df.index, ymin, ymax, 
                         where=(detection_df['ground_truth'] == 1), 
                         color='red', alpha=0.3, interpolate=True)
        gt_changes = detection_df['ground_truth'].diff()
        for idx in detection_df.index[gt_changes == 1]:
            ax3.axvline(idx, color='red', linestyle='--', linewidth=1)
        for idx in detection_df.index[gt_changes == -1]:
            ax3.axvline(idx, color='red', linestyle='--', linewidth=1)
        ax3.set_title('Energy Score Over Time')
        ax3.set_ylabel('Energy Score')
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.set_ylim(ymin, ymax)

        # --- Subplot 4: Energy Score Distribution (KDE) ---
        # This plot does not share the x-axis (time)
        fig.delaxes(axes[3]) # Remove the shared-x axis
        ax4 = fig.add_subplot(4, 1, 4)
        palette = {0: 'steelblue', 1: 'red'}
        sns.kdeplot(data=detection_df, x='score', hue='ground_truth', fill=False, 
                    common_norm=False, palette=palette, ax=ax4, legend=False)
        ax4.set_title('Energy Score Distribution')
        ax4.set_xlabel('Energy Score')
        ax4.set_ylabel('Density')
        ax4.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Combined plot saved to {output_filename}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate a combined plot for time series anomaly detection analysis.')
    parser.add_argument('--csv-path', required=True, help='Path to the detection results CSV file.')
    parser.add_argument('--training-data-path', required=True, help='Path to the Training.mat file.')
    parser.add_argument('--faulty-data-path', required=True, help='Path to the faulty case .mat file.')
    parser.add_argument('--normal-set', required=True, help="Name of the normal data set (e.g., 'T1').")
    parser.add_argument('--faulty-case', required=True, help="Name of the faulty case (e.g., 'FaultyCase5').")
    parser.add_argument('--faulty-set', required=True, help="Name of the specific faulty set (e.g., 'Set5_2').")
    parser.add_argument('--feature-idx', type=int, default=0, help='Index of the feature to plot.')
    parser.add_argument('--output', default='./results/plots/combined_plot.png', help='Path to save the output plot.')
    args = parser.parse_args()

    # Load detection results data
    try:
        detection_df = pd.read_csv(args.csv_path, index_col=0)
        detection_df.index.name = 'index'
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_path}")
        return

    # Load normal and faulty signal data
    normal_data = load_normal_data(Path(args.training_data_path), args.normal_set)
    faulty_data = load_faulty_data(Path(args.faulty_data_path), args.faulty_set)

    if normal_data is not None and faulty_data is not None:
        # Ensure output directory exists
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        plot_combined(
            normal_data=normal_data,
            faulty_data=faulty_data,
            detection_df=detection_df,
            normal_label=args.normal_set,
            faulty_case_name=args.faulty_case,
            faulty_set_label=args.faulty_set,
            feature_idx=args.feature_idx,
            output_filename=args.output
        )

if __name__ == '__main__':
    main()

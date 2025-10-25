import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

# Definition of fault intervals for each faulty case and set.
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
    """Loads a specific normal dataset (T1, T2, or T3) from the Training.mat file."""
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
        # The last feature is often not needed, so we remove it.
        return np.delete(data[faulty_set_name], -1, axis=1)
    except FileNotFoundError:
        print(f"Error: Faulty case file not found at {file_path}")
        return None
    except KeyError:
        print(f"Error: Faulty set '{faulty_set_name}' not found in {file_path.name}.")
        return None

def plot_comparison(normal_data, faulty_data, normal_label, faulty_case_name, faulty_set_label, feature_idx):
    """
    Generates a side-by-side plot of normal and faulty data for a specific feature,
    with the faulty region highlighted.
    """
    # Set larger font sizes for all plot elements within this context
    with plt.rc_context({'font.size': 14,
                         'axes.labelsize': 16,
                         'axes.titlesize': 18,
                         'xtick.labelsize': 14,
                         'ytick.labelsize': 14,
                         'figure.titlesize': 20}):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

        # Left plot: Normal Data
        ax1.plot(normal_data[:, feature_idx], color='black', linewidth=1.5)
        # ax1.set_title(f'Normal Data: {normal_label} - Feature {feature_idx + 1}', fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.grid(True, linestyle='--', alpha=0.5)
        # ax1.legend([normal_label], loc='upper right')

        # Right plot: Faulty Data
        ax2.plot(faulty_data[:, feature_idx], color='steelblue', linewidth=1.5)
        # ax2.set_title(f'Faulty Data: {faulty_case_name} {faulty_set_label} - Feature {feature_idx + 1}', fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.grid(True, linestyle='--', alpha=0.5)
        # ax2.legend([faulty_set_label], loc='upper right')

        # Highlight the fault intervals on the faulty data plot
        intervals = fault_intervals.get(faulty_case_name, {}).get(faulty_set_label, [])
        for start, end in intervals:
            ax2.axvspan(start, end, color='red', alpha=0.3, label='Faulty Region')
            ax2.axvline(start, color='red', linestyle='--', linewidth=1)
            ax2.axvline(end, color='red', linestyle='--', linewidth=1)
        
        # Ensure the legend for the fault region doesn't have duplicate labels
        # handles, labels = ax2.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax2.legend(by_label.values(), by_label.keys(), loc='upper right')

        # fig.suptitle(f'Comparison of Feature {feature_idx + 1}', y=1.02)
        plt.tight_layout()

        # Create a descriptive filename
        save_dir = Path("./results/plots")
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"comparison_F{feature_idx + 1}_{normal_label}_vs_{faulty_case_name}_{faulty_set_label}.png"

        # Save the figure
        plt.savefig(filename, bbox_inches='tight')
        print(f"Plot saved to {filename}")
        plt.close()  # Close the plot to free up memory

if __name__ == "__main__":
    # --- Configuration ---
    # Choose the feature index to plot (0 to 22)
    FEATURE_TO_PLOT = 0
    # Choose the normal data set ('T1', 'T2', or 'T3')
    NORMAL_SET_TO_PLOT = 'T2'
    # Choose the faulty case ('FaultyCase1' through 'FaultyCase5')
    FAULTY_CASE_TO_PLOT = 'FaultyCase2'
    # Choose the specific set within the faulty case
    FAULTY_SET_TO_PLOT = 'Set2_1'
    # --- End of Configuration ---

    # Define the base directory for datasets
    data_dir = Path("./Datasets/CVACaseStudy/MFP")

    # Load normal data
    training_file_path = data_dir / "Training.mat"
    normal_data = load_normal_data(training_file_path, NORMAL_SET_TO_PLOT)

    # Load faulty data
    faulty_file_path = data_dir / f"{FAULTY_CASE_TO_PLOT}.mat"
    faulty_data = load_faulty_data(faulty_file_path, FAULTY_SET_TO_PLOT)

    # Plot if data was loaded successfully
    if normal_data is not None and faulty_data is not None:
        plot_comparison(
            normal_data=normal_data,
            faulty_data=faulty_data,
            normal_label=NORMAL_SET_TO_PLOT,
            faulty_case_name=FAULTY_CASE_TO_PLOT,
            faulty_set_label=FAULTY_SET_TO_PLOT,
            feature_idx=FEATURE_TO_PLOT
        )


from pathlib import Path
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

def load_all_normal_data(file_path: Path):
    """Loads all normal datasets (T1, T2, T3) from the Training.mat file."""
    data_dict = {}
    try:
        mat_data = scio.loadmat(file_path)
        for name in ['T1', 'T2', 'T3']:
            if name in mat_data:
                data_dict[name] = mat_data[name]
            else:
                print(f"Warning: Normal set '{name}' not found in the training file.")
    except FileNotFoundError:
        print(f"Error: Training file not found at {file_path}")
        return None
    return data_dict

def load_all_faulty_data(data_dir: Path, fault_intervals: dict):
    """Loads all faulty datasets defined in fault_intervals."""
    data_dict = {}
    for case_name, sets in fault_intervals.items():
        file_path = data_dir / f"{case_name}.mat"
        try:
            mat_data = scio.loadmat(file_path)
            for set_name in sets.keys():
                if set_name in mat_data:
                    # The last feature is often not needed, so we remove it.
                    data_dict[f"{case_name}_{set_name}"] = np.delete(mat_data[set_name], -1, axis=1)
                else:
                    print(f"Warning: Faulty set '{set_name}' not found in {file_path.name}.")
        except FileNotFoundError:
            print(f"Error: Faulty case file not found at {file_path}")
            continue
    return data_dict

def plot_all_data_grid(normal_data_dict, faulty_data_dict, feature_idx, fault_intervals):
    """
    Generates a grid plot of normal and faulty data for a specific feature.
    """
    num_normal = len(normal_data_dict)
    num_faulty = len(faulty_data_dict)
    total_plots = num_normal + num_faulty

    nrows = 6
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 24))
    axes = axes.flatten()

    plot_idx = 0

    # Plot normal data
    for name, data in normal_data_dict.items():
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            ax.plot(data[:, feature_idx], color='black', linewidth=1)
            ax.set_title(f'Normal: {name}', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            plot_idx += 1

    # Plot faulty data
    for name, data in faulty_data_dict.items():
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            parts = name.split('_')
            case_name = parts[0]
            set_name = '_'.join(parts[1:])
            
            ax.plot(data[:, feature_idx], color='steelblue', linewidth=1)
            ax.set_title(f'Faulty: {case_name} {set_name}', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)

            # Highlight fault intervals
            intervals = fault_intervals.get(case_name, {}).get(set_name, [])
            for start, end in intervals:
                ax.axvspan(start, end, color='red', alpha=0.3)
            
            plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Comparison for Feature {feature_idx + 1}', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_dir = Path("./results/plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"all_cases_feature_{feature_idx + 1}.png"

    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close()

if __name__ == "__main__":
    # --- Configuration ---
    # Choose the feature index to plot (0 to 22)
    FEATURE_TO_PLOT = 22
    # --- End of Configuration ---

    data_dir = Path("./Datasets/CVACaseStudy/MFP")
    training_file_path = data_dir / "Training.mat"

    normal_data = load_all_normal_data(training_file_path)
    faulty_data = load_all_faulty_data(data_dir, fault_intervals)

    if normal_data and faulty_data:
        plot_all_data_grid(
            normal_data_dict=normal_data,
            faulty_data_dict=faulty_data,
            feature_idx=FEATURE_TO_PLOT,
            fault_intervals=fault_intervals
        )

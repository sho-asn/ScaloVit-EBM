# make_plots.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Make sure you have run preprocess_data.py first to create these files.
from utils.utils_visualization import plot_scalogram_from_file

print("Creating scalogram plots...")

# --- Example 1: Plot a chunk from a specific test file ---
# This file contains chunks from a signal with known anomalies.
test_file_to_plot = 'preprocessed_dataset/test_FaultyCase1_Set1_1_wavelet.pt'
output_directory = 'results/plots/scalograms'

# Call the function to plot the 10th chunk and the 5th feature from that file
plot_scalogram_from_file(
    file_path=test_file_to_plot,
    save_dir=output_directory,
    chunk_index=10,
    feature_index=5
)

# --- Example 2: Plot a chunk from the training data ---
# This file contains chunks from the normal training signal.
train_file_to_plot = 'preprocessed_dataset/train_chunks_wavelet.pt'

# Call the function to plot the 100th training chunk and the first feature (index 0)
plot_scalogram_from_file(
    file_path=train_file_to_plot,
    save_dir=output_directory,
    chunk_index=100,
    feature_index=0
)

print("Done.")

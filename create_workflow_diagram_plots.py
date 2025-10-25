import numpy as np
import scipy.io as scio
from scipy.ndimage import uniform_filter1d
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

def create_workflow_plots():
    """
    Generates and saves a plot illustrating the workflow from raw signal
    to detrended signal to CWT scalogram, intended for use in a workflow diagram.
    """
    print("--- Generating Workflow Diagram Plots ---")

    # --- 1. Load a Sample Signal ---
    # This logic is adapted from preprocess_data.py
    data_dir = Path("Datasets/CVACaseStudy/MFP")
    training_data_path = data_dir / "Training.mat"
    
    try:
        data = scio.loadmat(training_data_path)
    except FileNotFoundError:
        print(f"Error: Training data not found at {training_data_path}")
        print("Please ensure the dataset is correctly placed in the 'Datasets' directory.")
        return

    # Use a 4096-step slice of the first feature from the T1 signal as an example
    raw_signal = np.delete(data['T1'], -1, axis=1)[:4096, 0]
    time_steps = np.arange(raw_signal.shape[0])
    print(f"Loaded a sample raw signal with shape: {raw_signal.shape}")

    # --- 2. Detrend the Signal ---
    # This uses the moving average method found in preprocess_data.py
    detrend_window_size = 256
    trend = uniform_filter1d(raw_signal, size=detrend_window_size, axis=0, mode='nearest')
    detrended_signal = raw_signal - trend
    print(f"Detrended signal using a moving average with window size {detrend_window_size}.")

    # --- 3. Perform Continuous Wavelet Transform (CWT) ---
    # This logic is adapted from the WAVEmbedder in img_transformations.py
    wavelet_name = 'morl'
    # Using scales similar to the default in preprocess_data.py (1 to 256)
    scales = np.arange(1, 257)
    
    coeffs, frequencies = pywt.cwt(detrended_signal, scales, wavelet_name)
    
    # The scalogram is the magnitude of the CWT coefficients
    scalogram = np.abs(coeffs)
    print(f"Performed CWT to generate a scalogram with shape: {scalogram.shape}")

    # --- 4. Create and Save Plots Separately ---
    output_dir = Path("results/plots")
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Raw Signal
    fig1, ax1 = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax1.plot(time_steps, raw_signal, color='royalblue')
    ax1.set_title("Step 1: Raw Signal", fontsize=12)
    ax1.set_xlabel("Time Step", fontsize=10)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    raw_signal_path = output_dir / "workflow_raw_signal.png"
    print(f"Saving raw signal plot to: {raw_signal_path}")
    plt.savefig(raw_signal_path, dpi=300)
    plt.close(fig1)

    # Plot 2: Detrended Signal
    fig2, ax2 = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax2.plot(time_steps, detrended_signal, color='green')
    ax2.set_title("Step 2: Detrended Signal (Residual)", fontsize=12)
    ax2.set_xlabel("Time Step", fontsize=10)
    ax2.set_ylabel("Amplitude", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    detrended_signal_path = output_dir / "workflow_detrended_signal.png"
    print(f"Saving detrended signal plot to: {detrended_signal_path}")
    plt.savefig(detrended_signal_path, dpi=300)
    plt.close(fig2)

    # Plot 3: CWT Scalogram
    fig3, ax3 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    im = ax3.imshow(scalogram, aspect='auto', cmap='turbo',
                    extent=[time_steps[0], time_steps[-1], frequencies[-1], frequencies[0]])
    ax3.set_title("Step 3: CWT Scalogram Image", fontsize=12)
    ax3.set_xlabel("Time Step", fontsize=10)
    ax3.set_ylabel("Frequency (Derived from Scale)", fontsize=10)
    cbar = fig3.colorbar(im, ax=ax3)
    cbar.set_label("Magnitude", fontsize=10)
    scalogram_path = output_dir / "workflow_scalogram.png"
    print(f"Saving scalogram plot to: {scalogram_path}")
    plt.savefig(scalogram_path, dpi=300)
    plt.close(fig3)

    # Plot 4: Scalogram with Chunking Visualization
    chunk_size = 1024
    num_chunks = raw_signal.shape[0] // chunk_size
    
    fig4, ax4 = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax4.imshow(scalogram, aspect='auto', cmap='turbo',
               extent=[time_steps[0], time_steps[-1], frequencies[-1], frequencies[0]])
    
    # Draw vertical lines for chunk boundaries
    for i in range(1, num_chunks):
        ax4.axvline(x=i * chunk_size, color='r', linestyle='--', linewidth=2.5)

    # Add text labels for each chunk
    for i in range(num_chunks):
        ax4.text(i * chunk_size + chunk_size / 2, scalogram.shape[0] / 2, f'Chunk {i+1}',
                 color='white', ha='center', va='center', fontsize=14, weight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))

    ax4.set_title(f"Step 4: Scalogram Chunking (Chunk Size = {chunk_size})", fontsize=12)
    ax4.set_xlabel("Time Step", fontsize=10)
    ax4.set_ylabel("Frequency (Derived from Scale)", fontsize=10)
    
    chunking_path = output_dir / "workflow_scalogram_chunking.png"
    print(f"Saving chunking visualization plot to: {chunking_path}")
    plt.savefig(chunking_path, dpi=300)
    plt.close(fig4)

    print("--- All plot generation complete. ---")


if __name__ == "__main__":
    create_workflow_plots()

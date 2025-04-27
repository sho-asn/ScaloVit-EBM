import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Optional, Union
from matplotlib.backends.backend_pdf import PdfPages


def plot_signal_and_spectrograms(
        signal: torch.Tensor,
        spectrogram: torch.Tensor,
        batch_idx: int = 0,
        channel_idx: int = 0
        ) -> plt.Figure:
    """
    Create a figure showing signal, real part, and imaginary part spectrograms.

    Args:
        signal: Shape (B, L, F)
        spectrogram: Shape (B, 2*F, Freq, Time)
        batch_idx: Batch index
        channel_idx: Channel (feature) index
    
    Returns:
        plt.Figure: Matplotlib figure object
    """
    signal = signal.cpu().detach()
    spectrogram = spectrogram.cpu().detach()

    B, L, F = signal.shape
    _, CF, Freq, Time = spectrogram.shape
    assert CF == 2 * F, f"Expected spectrogram channels to be 2x features. Got {CF} != 2*{F}"

    real_idx = channel_idx * 2
    imag_idx = channel_idx * 2 + 1

    real_spectrogram = spectrogram[batch_idx, real_idx]
    imag_spectrogram = spectrogram[batch_idx, imag_idx]
    selected_signal = signal[batch_idx, :, channel_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Plot signal
    axes[0].plot(selected_signal.numpy())
    axes[0].set_title(f"Signal (Batch {batch_idx}, Feature {channel_idx})")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Plot real part
    im1 = axes[1].imshow(real_spectrogram.numpy(), aspect='auto', origin='lower')
    axes[1].set_title("Real Part Spectrogram")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency Bin")
    plt.colorbar(im1, ax=axes[1])

    # Plot imaginary part
    im2 = axes[2].imshow(imag_spectrogram.numpy(), aspect='auto', origin='lower')
    axes[2].set_title("Imaginary Part Spectrogram")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Frequency Bin")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    return fig


def save_all_plots_to_pdf(
        signal: torch.Tensor,
        spectrogram: torch.Tensor,
        pdf_path: Union[str, Path],
        dpi: int = 300
        ) -> None:
    """
    Save all batch/feature plots into a single multi-page PDF.

    Args:
        signal: Shape (B, L, F)
        spectrogram: Shape (B, 2*F, Freq, Time)
        pdf_path: Output PDF file path
        dpi: DPI resolution
    """
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        B, L, F = signal.shape
        for batch_idx in range(B):
            for channel_idx in range(F):
                fig = plot_signal_and_spectrograms(signal, spectrogram, batch_idx, channel_idx)
                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)

    print(f"Saved all plots to {pdf_path}")
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Optional, Union, List, Tuple
from matplotlib.backends.backend_pdf import PdfPages


def plot_signal_and_spectrograms(
        signal: torch.Tensor,
        spectrogram: torch.Tensor,
        batch_idx: int,
        feature_idx: int,
        signal_xlim: Optional[Tuple[int, int]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
        ) -> plt.Figure:
    """
    Create a figure showing signal, real part, and imaginary part spectrograms for one batch and one feature.

    Args:
        signal: Tensor of shape (B, L, F)
        spectrogram: Tensor of shape (B, 2*F, Freq, Time)
        batch_idx: Batch index
        feature_idx: Feature index
        signal_xlim: Optional (start_idx, end_idx) to limit x-axis of signal plot.
        vmin: Optional minimum value for spectrogram color scale.
        vmax: Optional maximum value for spectrogram color scale.

    Returns:
        plt.Figure
    """
    signal = signal.cpu().detach()
    spectrogram = spectrogram.cpu().detach()

    B, L, F = signal.shape
    _, CF, Freq, Time = spectrogram.shape
    assert CF == 2 * F, f"Expected spectrogram channels to be 2x features. Got {CF} != 2*{F}"

    real_idx = feature_idx
    imag_idx = feature_idx + F
    real_spectrogram = spectrogram[batch_idx, real_idx]
    imag_spectrogram = spectrogram[batch_idx, imag_idx]
    selected_signal = signal[batch_idx, :, feature_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot signal
    axes[0].plot(selected_signal.numpy())
    axes[0].set_title(f"Signal (Batch {batch_idx}, Feature {feature_idx})")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)
    if signal_xlim is not None:
        axes[0].set_xlim(signal_xlim)

    # Plot real part
    im1 = axes[1].imshow(real_spectrogram.numpy(), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title("Real Part Spectrogram")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency Bin")
    plt.colorbar(im1, ax=axes[1])

    # Plot imaginary part
    im2 = axes[2].imshow(imag_spectrogram.numpy(), aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title("Imaginary Part Spectrogram")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Frequency Bin")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    return fig


def save_feature_plots_to_pdf(
        signal: torch.Tensor,
        spectrogram: torch.Tensor,
        feature_idx: int,
        pdf_path: Union[str, Path],
        batch_indices: Optional[List[int]] = None,
        signal_xlim: Optional[Tuple[int, int]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        dpi: int = 300
        ) -> None:
    """
    Save signal + real/imaginary spectrogram plots for selected feature to a single multi-page PDF.

    Args:
        signal: Tensor of shape (B, L, F)
        spectrogram: Tensor of shape (B, 2*F, Freq, Time)
        feature_idx: Feature index to plot
        pdf_path: Output PDF file path
        batch_indices: Optional list of batch indices to plot (default: all batches)
        signal_xlim: Optional (start_idx, end_idx) for signal plot x-axis range
        vmin: Optional minimum value for spectrogram color scale.
        vmax: Optional maximum value for spectrogram color scale.
        dpi: DPI resolution for saving
    """
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    signal = signal.cpu()
    spectrogram = spectrogram.cpu()

    B, _, _ = signal.shape
    if batch_indices is None:
        batch_indices = list(range(B))

    with PdfPages(pdf_path) as pdf:
        for batch_idx in batch_indices:
            fig = plot_signal_and_spectrograms(signal, spectrogram, batch_idx, feature_idx, signal_xlim, vmin, vmax)
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    print(f"Saved {len(batch_indices)} plots for feature {feature_idx} to {pdf_path}")
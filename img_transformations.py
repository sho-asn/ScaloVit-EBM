from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import pywt
import numpy as np
from scipy.signal import stft
from utils.utils_data import MinMaxArgs


class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images.
    """

    def __init__(self, device: torch.device, seq_len: int):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: given time series
        Returns:
            image representation of the signal
        """
        pass

    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: given generated image
        Returns:
            time series representation of the generated image
        """
        raise NotImplementedError("Reconstruction is not supported by this embedder.")


class WAVEmbedder(TsImgEmbedder):
    """
    Transforms a time series into a 2-channel image using Continuous Wavelet Transform.
    The two channels represent the normalized magnitude (scalogram) and phase of the CWT.
    """
    def __init__(self, 
                 device: torch.device, 
                 seq_len: int, 
                 wavelet_name: str = 'morl', 
                 scales_arange: Tuple[int, int] = (1, 129)):
        super().__init__(device, seq_len)
        self.wavelet_name: str = wavelet_name
        self.scales: np.ndarray = np.arange(scales_arange[0], scales_arange[1])
        self.min_mag: Optional[torch.Tensor] = None
        self.max_mag: Optional[torch.Tensor] = None
        self.min_phase: Optional[torch.Tensor] = None
        self.max_phase: Optional[torch.Tensor] = None

    def cache_min_max_params(self, train_data: np.ndarray) -> None:
        """
        Calculates and caches the min/max of CWT magnitude and phase for each feature.
        """
        real, imag = self._wav_transform_raw(train_data)
        
        magnitude = np.sqrt(real**2 + imag**2)
        phase = np.arctan2(imag, real)
        
        num_features = magnitude.shape[1]

        self.min_mag = torch.tensor([np.min(magnitude[:, k, :, :]) for k in range(num_features)])
        self.max_mag = torch.tensor([np.max(magnitude[:, k, :, :]) for k in range(num_features)])
        self.min_phase = torch.tensor([np.min(phase[:, k, :, :]) for k in range(num_features)])
        self.max_phase = torch.tensor([np.max(phase[:, k, :, :]) for k in range(num_features)])

    def _wav_transform_raw(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Performs the CWT and returns the raw real and imaginary coefficients. """
        num_batches, seq_len, num_features = data.shape

        batch_coeffs = []
        for b in range(num_batches):
            feature_coeffs = []
            for k in range(num_features):
                signal = data[b, :, k]
                coeffs, _ = pywt.cwt(signal, self.scales, self.wavelet_name)
                feature_coeffs.append(coeffs)
            stacked_features = np.stack(feature_coeffs, axis=0)
            batch_coeffs.append(stacked_features)
        
        full_coeffs = np.stack(batch_coeffs, axis=0)

        return full_coeffs.real, full_coeffs.imag

    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Transforms the signal into a 2-channel image of normalized magnitude and phase.
        """
        assert self.min_mag is not None, "Normalization parameters not cached. Call `cache_min_max_params` first."

        # Get raw CWT coefficients
        real, imag = self._wav_transform_raw(signal.cpu().numpy())
        
        # Calculate magnitude and phase
        magnitude = torch.tensor(np.sqrt(real**2 + imag**2), device=self.device)
        phase = torch.tensor(np.arctan2(imag, real), device=self.device)

        # Reshape scalers for broadcasting: (1, K, 1, 1)
        min_mag = self.min_mag.view(1, -1, 1, 1).to(self.device)
        max_mag = self.max_mag.view(1, -1, 1, 1).to(self.device)
        min_phase = self.min_phase.view(1, -1, 1, 1).to(self.device)
        max_phase = self.max_phase.view(1, -1, 1, 1).to(self.device)

        # Normalize magnitude and phase independently
        norm_magnitude = (MinMaxArgs(magnitude, min_mag, max_mag) - 0.5) * 2
        norm_phase = (MinMaxArgs(phase, min_phase, max_phase) - 0.5) * 2

        # The output channels for each feature will be [mag, phase]
        # We stack and reshape to achieve this interleaving.
        # Start with shape (B, K, 2, S, L) then reshape to (B, 2*K, S, L)
        num_batches, num_features, num_scales, num_len = norm_magnitude.shape
        
        output_image = torch.stack([norm_magnitude, norm_phase], dim=2)
        output_image = output_image.view(num_batches, 2 * num_features, num_scales, num_len)
        
        return output_image

def init_wav_embedder(embedder: "WAVEmbedder", full_train_signal_np: np.ndarray) -> None:
    """
    Initializes min/max values for normalization across the whole training signal.
    Args:
        embedder (WAVEmbedder): the embedder object.
        full_train_signal_np (np.ndarray): The entire training time-series data as a numpy array.
                                             Shape: (1, L, F) for a single batch of the full signal.
    """
    embedder.cache_min_max_params(full_train_signal_np)

def split_image_into_chunks(image: torch.Tensor, chunk_width: int) -> torch.Tensor:
    """
    Splits a 4D image tensor (B, C, H, W) into chunks along the Width (W) dimension.

    Args:
        image (torch.Tensor): The input 4D tensor representing the wavelet image.
                              Shape: (B, C, H, W) where B is batch, C is channels,
                              H is height (scales), W is width (time steps).
        chunk_width (int): The desired width for each chunk.

    Returns:
        torch.Tensor: A new tensor containing the image chunks.
                      Shape: (B * num_chunks, C, H, chunk_width).
    """
    batch_size, channels, height, width = image.shape
    num_chunks = width // chunk_width

    if num_chunks == 0:
        return torch.empty(0, channels, height, chunk_width)

    # Ensure the image width is a multiple of chunk_width by truncating if necessary
    truncated_width = num_chunks * chunk_width
    image = image[:, :, :, :truncated_width]

    # Reshape: (B, C, H, num_chunks * chunk_width) -> (B, C, H, num_chunks, chunk_width)
    # Then permute and reshape to get (B * num_chunks, C, H, chunk_width)
    chunks = image.reshape(batch_size, channels, height, num_chunks, chunk_width)
    chunks = chunks.permute(0, 3, 1, 2, 4).contiguous() # Reorder to (B, num_chunks, C, H, chunk_width)
    chunks = chunks.view(-1, channels, height, chunk_width) # Flatten B and num_chunks

    return chunks


def split_image_into_chunks_with_stride(image: torch.Tensor, chunk_width: int, stride: int) -> torch.Tensor:
    """
    Splits a 4D image tensor (B, C, H, W) into chunks along the Width (W) dimension using a sliding window.
    This implementation uses torch.unfold for efficiency and handles batch processing correctly.

    Args:
        image (torch.Tensor): The input 4D tensor. Shape: (B, C, H, W).
        chunk_width (int): The width of each chunk.
        stride (int): The step size to move the window.

    Returns:
        torch.Tensor: A new tensor containing the image chunks.
                      Shape: (num_total_chunks, C, H, chunk_width), where
                      num_total_chunks is the total number of chunks from all images in the batch.
    """
    batch_size, channels, height, width = image.shape

    # Use unfold to create sliding window views from the images.
    # The result is (B, C, H, num_chunks, chunk_width)
    unfolded = image.unfold(3, chunk_width, stride)

    # Permute to bring num_chunks next to the batch dimension: (B, num_chunks, C, H, chunk_width)
    unfolded = unfolded.permute(0, 3, 1, 2, 4)

    # Reshape to combine batch and chunk dimensions: (B * num_chunks, C, H, chunk_width)
    chunks = unfolded.reshape(-1, channels, height, chunk_width)

    return chunks


class STFTEmbedder(TsImgEmbedder):
    """
    Transforms a time series into a 2-channel image using Short-Time Fourier Transform (STFT).
    The two channels represent the normalized magnitude and phase of the STFT.
    """
    def __init__(self,
                 device: torch.device,
                 seq_len: int,
                 nperseg: int = 256,
                 noverlap: int = 128,
                 nfft: int = 256):
        super().__init__(device, seq_len)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.min_mag: Optional[torch.Tensor] = None
        self.max_mag: Optional[torch.Tensor] = None
        self.min_phase: Optional[torch.Tensor] = None
        self.max_phase: Optional[torch.Tensor] = None

    def cache_min_max_params(self, train_data: np.ndarray) -> None:
        """
        Calculates and caches the min/max of STFT magnitude and phase for each feature.
        """
        _, _, Zxx = self._stft_transform_raw(train_data)
        
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        num_features = magnitude.shape[1]

        self.min_mag = torch.tensor([np.min(magnitude[:, k, :, :]) for k in range(num_features)])
        self.max_mag = torch.tensor([np.max(magnitude[:, k, :, :]) for k in range(num_features)])
        self.min_phase = torch.tensor([np.min(phase[:, k, :, :]) for k in range(num_features)])
        self.max_phase = torch.tensor([np.max(phase[:, k, :, :]) for k in range(num_features)])

    def _stft_transform_raw(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Performs the STFT and returns the raw frequencies, times, and complex values. """
        num_batches, seq_len, num_features = data.shape

        batch_Zxx = []
        for b in range(num_batches):
            feature_Zxx = []
            for k in range(num_features):
                signal = data[b, :, k]
                f, t, Zxx = stft(signal, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
                feature_Zxx.append(Zxx)
            stacked_features = np.stack(feature_Zxx, axis=0)
            batch_Zxx.append(stacked_features)
        
        full_Zxx = np.stack(batch_Zxx, axis=0)

        # For STFT, f and t are the same for all batches and features
        f, t, _ = stft(data[0, :, 0], nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
        return f, t, full_Zxx

    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Transforms the signal into a 2-channel image of normalized magnitude and phase.
        """
        assert self.min_mag is not None, "Normalization parameters not cached. Call `cache_min_max_params` first."

        # Get raw STFT coefficients
        _, _, Zxx = self._stft_transform_raw(signal.cpu().numpy())
        
        # Calculate magnitude and phase
        magnitude = torch.tensor(np.abs(Zxx), device=self.device)
        phase = torch.tensor(np.angle(Zxx), device=self.device)

        # Reshape scalers for broadcasting: (1, K, 1, 1)
        min_mag = self.min_mag.view(1, -1, 1, 1).to(self.device)
        max_mag = self.max_mag.view(1, -1, 1, 1).to(self.device)
        min_phase = self.min_phase.view(1, -1, 1, 1).to(self.device)
        max_phase = self.max_phase.view(1, -1, 1, 1).to(self.device)

        # Normalize magnitude and phase independently
        norm_magnitude = (MinMaxArgs(magnitude, min_mag, max_mag) - 0.5) * 2
        norm_phase = (MinMaxArgs(phase, min_phase, max_phase) - 0.5) * 2

        # The output channels for each feature will be [mag, phase]
        num_batches, num_features, num_freqs, num_times = norm_magnitude.shape
        
        output_image = torch.stack([norm_magnitude, norm_phase], dim=2)
        output_image = output_image.view(num_batches, 2 * num_features, num_freqs, num_times)
        
        return output_image

def init_stft_embedder(embedder: "STFTEmbedder", full_train_signal_np: np.ndarray) -> None:
    """
    Initializes min/max values for normalization across the whole training signal.
    Args:
        embedder (STFTEmbedder): the embedder object.
        full_train_signal_np (np.ndarray): The entire training time-series data as a numpy array.
                                             Shape: (1, L, F) for a single batch of the full signal.
    """
    embedder.cache_min_max_params(full_train_signal_np)
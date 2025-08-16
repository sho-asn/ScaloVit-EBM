from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import pywt
import numpy as np
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
                 scales_arange: Tuple[int, int] = (1, 128)):
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

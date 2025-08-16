from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import pywt
import numpy as np
from utils.utils_data import MinMaxArgs


class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
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

    @abstractmethod
    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass


class WAVEmbedder(TsImgEmbedder):
    def __init__(self, 
                 device: torch.device, 
                 seq_len: int, 
                 wavelet_name: str = 'morl', 
                 scales_arange: Tuple[int, int] = (1, 64)):
        super().__init__(device, seq_len)
        self.wavelet_name: str = wavelet_name
        self.scales: np.ndarray = np.arange(scales_arange[0], scales_arange[1])
        self.min_real: Optional[torch.Tensor] = None
        self.max_real: Optional[torch.Tensor] = None
        self.min_imag: Optional[torch.Tensor] = None
        self.max_imag: Optional[torch.Tensor] = None

    def cache_min_max_params(self, train_data: np.ndarray) -> None:
        real, imag = self.wav_transform(train_data)
        num_features = real.shape[1]

        min_reals, max_reals = [], []
        min_imags, max_imags = [], []

        for k in range(num_features):
            real_k = real[:, k, :, :]
            min_reals.append(np.min(real_k))
            max_reals.append(np.max(real_k))

            imag_k = imag[:, k, :, :]
            min_imags.append(np.min(imag_k))
            max_imags.append(np.max(imag_k))

        self.min_real = torch.tensor(min_reals)
        self.max_real = torch.tensor(max_reals)
        self.min_imag = torch.tensor(min_imags)
        self.max_imag = torch.tensor(max_imags)

    def wav_transform(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # data is a numpy array of shape (B, L, K)
        num_batches, seq_len, num_features = data.shape
        
        all_reals = []
        all_imags = []

        for k in range(num_features):
            feature_data = data[:, :, k] # Shape (B, L)
            coeffs, _ = pywt.cwt(feature_data, self.scales, self.wavelet_name, axis=1)
            # coeffs shape is (B, num_scales, L)
            all_reals.append(coeffs.real)
            all_imags.append(coeffs.imag)

        # all_reals is a list of K arrays of shape (B, S, L)
        # stack them on feature dimension
        real_coeffs = np.stack(all_reals, axis=1) # (B, K, S, L)
        imag_coeffs = np.stack(all_imags, axis=1) # (B, K, S, L)

        return real_coeffs, imag_coeffs

    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        # signal is a torch tensor (B, L, K)
        assert self.min_real is not None, "use cache_min_max_params() to compute scaling arguments"

        real, imag = self.wav_transform(signal.cpu().numpy())
        real, imag = torch.Tensor(real).to(self.device), torch.Tensor(imag).to(self.device)

        # self.min/max_real/imag are of shape (K,)
        # Reshape for broadcasting: (1, K, 1, 1)
        min_real = self.min_real.view(1, -1, 1, 1).to(self.device)
        max_real = self.max_real.view(1, -1, 1, 1).to(self.device)
        min_imag = self.min_imag.view(1, -1, 1, 1).to(self.device)
        max_imag = self.max_imag.view(1, -1, 1, 1).to(self.device)

        # MinMax scaling per feature
        real = (MinMaxArgs(real, min_real, max_real) - 0.5) * 2
        imag = (MinMaxArgs(imag, min_imag, max_imag) - 0.5) * 2

        wavelet_out = torch.cat((real, imag), dim=1)
        return wavelet_out

    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Reconstruction from image to time series is not supported.")
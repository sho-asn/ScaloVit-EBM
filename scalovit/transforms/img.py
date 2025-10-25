"""Time-series to image transformation utilities (CWT/STFT)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pywt
import torch
from scipy.signal import stft

from scalovit.utils.data import MinMaxArgs


class TsImgEmbedder(ABC):
    """Abstract base class for converting time series to image representations."""

    def __init__(self, device: torch.device, seq_len: int) -> None:
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        """Convert a time series into an image tensor."""

    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Reconstruction is not supported by this embedder.")


class WAVEmbedder(TsImgEmbedder):
    """Convert time series to scalograms using the continuous wavelet transform."""

    def __init__(
        self,
        device: torch.device,
        seq_len: int,
        wavelet_name: str = "morl",
        scales_arange: Tuple[int, int] = (1, 129),
    ) -> None:
        super().__init__(device, seq_len)
        self.wavelet_name = wavelet_name
        self.scales = np.arange(scales_arange[0], scales_arange[1])
        self.min_mag: torch.Tensor | None = None
        self.max_mag: torch.Tensor | None = None
        self.min_phase: torch.Tensor | None = None
        self.max_phase: torch.Tensor | None = None

    def cache_min_max_params(self, train_data_list: list[np.ndarray]) -> None:
        if not train_data_list:
            return

        num_features = train_data_list[0].shape[1]
        self.min_mag = torch.full((num_features,), float("inf"))
        self.max_mag = torch.full((num_features,), float("-inf"))
        self.min_phase = torch.full((num_features,), float("inf"))
        self.max_phase = torch.full((num_features,), float("-inf"))

        for signal_np in train_data_list:
            if signal_np.shape[0] == 0:
                continue

            real, imag = self._wav_transform_raw(np.expand_dims(signal_np, axis=0))
            magnitude = np.sqrt(real**2 + imag**2)
            phase = np.arctan2(imag, real)

            for feature_idx in range(num_features):
                self.min_mag[feature_idx] = min(self.min_mag[feature_idx], np.min(magnitude[:, feature_idx, :, :]))
                self.max_mag[feature_idx] = max(self.max_mag[feature_idx], np.max(magnitude[:, feature_idx, :, :]))
                self.min_phase[feature_idx] = min(self.min_phase[feature_idx], np.min(phase[:, feature_idx, :, :]))
                self.max_phase[feature_idx] = max(self.max_phase[feature_idx], np.max(phase[:, feature_idx, :, :]))

    def _wav_transform_raw(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if data.ndim > 3:
            data = data.reshape(-1, data.shape[-2], data.shape[-1])
        num_batches, _, num_features = data.shape

        batch_coeffs = []
        for batch_idx in range(num_batches):
            feature_coeffs = []
            for feature_idx in range(num_features):
                signal = data[batch_idx, :, feature_idx]
                coeffs, _ = pywt.cwt(signal, self.scales, self.wavelet_name)
                feature_coeffs.append(coeffs)
            batch_coeffs.append(np.stack(feature_coeffs, axis=0))

        full_coeffs = np.stack(batch_coeffs, axis=0)
        return full_coeffs.real, full_coeffs.imag

    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        assert self.min_mag is not None, "Normalization parameters not cached. Call `cache_min_max_params` first."

        real, imag = self._wav_transform_raw(signal.cpu().numpy())
        magnitude = torch.tensor(np.sqrt(real**2 + imag**2), device=self.device)
        phase = torch.tensor(np.arctan2(imag, real), device=self.device)

        min_mag = self.min_mag.view(1, -1, 1, 1).to(self.device)
        max_mag = self.max_mag.view(1, -1, 1, 1).to(self.device)
        min_phase = self.min_phase.view(1, -1, 1, 1).to(self.device)
        max_phase = self.max_phase.view(1, -1, 1, 1).to(self.device)

        norm_magnitude = (MinMaxArgs(magnitude, min_mag, max_mag) - 0.5) * 2
        norm_phase = (MinMaxArgs(phase, min_phase, max_phase) - 0.5) * 2

        num_batches, num_features, num_scales, num_len = norm_magnitude.shape
        output_image = torch.stack([norm_magnitude, norm_phase], dim=2)
        output_image = output_image.view(num_batches, 2 * num_features, num_scales, num_len)
        return output_image


def init_wav_embedder(embedder: WAVEmbedder, train_signal_list: list[np.ndarray]) -> None:
    embedder.cache_min_max_params(train_signal_list)


def split_image_into_chunks(image: torch.Tensor, chunk_width: int) -> torch.Tensor:
    batch_size, channels, height, width = image.shape
    num_chunks = width // chunk_width
    if num_chunks == 0:
        return torch.empty(0, channels, height, chunk_width)

    truncated_width = num_chunks * chunk_width
    image = image[:, :, :, :truncated_width]
    chunks = image.reshape(batch_size, channels, height, num_chunks, chunk_width)
    chunks = chunks.permute(0, 3, 1, 2, 4).contiguous()
    chunks = chunks.view(-1, channels, height, chunk_width)
    return chunks


def split_image_into_chunks_with_stride(image: torch.Tensor, chunk_width: int, stride: int) -> torch.Tensor:
    batch_size, channels, height, width = image.shape
    if width < chunk_width:
        return torch.empty(0, channels, height, chunk_width, device=image.device)

    unfolded = image.unfold(3, chunk_width, stride)
    unfolded = unfolded.permute(0, 3, 1, 2, 4)
    chunks = unfolded.reshape(-1, channels, height, chunk_width)
    return chunks


class STFTEmbedder(TsImgEmbedder):
    """Convert time series to spectrogram-like images using the STFT."""

    def __init__(
        self,
        device: torch.device,
        seq_len: int,
        nperseg: int = 256,
        noverlap: int = 128,
        nfft: int = 256,
    ) -> None:
        super().__init__(device, seq_len)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.min_mag: torch.Tensor | None = None
        self.max_mag: torch.Tensor | None = None
        self.min_phase: torch.Tensor | None = None
        self.max_phase: torch.Tensor | None = None

    def cache_min_max_params(self, train_data: np.ndarray) -> None:
        _, _, zxx = self._stft_transform_raw(train_data)
        magnitude = np.abs(zxx)
        phase = np.angle(zxx)
        num_features = magnitude.shape[1]

        self.min_mag = torch.tensor([np.min(magnitude[:, idx, :, :]) for idx in range(num_features)])
        self.max_mag = torch.tensor([np.max(magnitude[:, idx, :, :]) for idx in range(num_features)])
        self.min_phase = torch.tensor([np.min(phase[:, idx, :, :]) for idx in range(num_features)])
        self.max_phase = torch.tensor([np.max(phase[:, idx, :, :]) for idx in range(num_features)])

    def _stft_transform_raw(self, data: np.ndarray):
        num_batches, _, num_features = data.shape
        batch_zxx = []
        for batch_idx in range(num_batches):
            feature_zxx = []
            for feature_idx in range(num_features):
                signal = data[batch_idx, :, feature_idx]
                _, _, zxx = stft(signal, nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
                feature_zxx.append(zxx)
            batch_zxx.append(np.stack(feature_zxx, axis=0))

        full_zxx = np.stack(batch_zxx, axis=0)
        freqs, times, _ = stft(data[0, :, 0], nperseg=self.nperseg, noverlap=self.noverlap, nfft=self.nfft)
        return freqs, times, full_zxx

    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        assert self.min_mag is not None, "Normalization parameters not cached. Call `cache_min_max_params` first."

        _, _, zxx = self._stft_transform_raw(signal.cpu().numpy())
        magnitude = torch.tensor(np.abs(zxx), device=self.device)
        phase = torch.tensor(np.angle(zxx), device=self.device)

        min_mag = self.min_mag.view(1, -1, 1, 1).to(self.device)
        max_mag = self.max_mag.view(1, -1, 1, 1).to(self.device)
        min_phase = self.min_phase.view(1, -1, 1, 1).to(self.device)
        max_phase = self.max_phase.view(1, -1, 1, 1).to(self.device)

        norm_magnitude = (MinMaxArgs(magnitude, min_mag, max_mag) - 0.5) * 2
        norm_phase = (MinMaxArgs(phase, min_phase, max_phase) - 0.5) * 2

        num_batches, num_features, num_freqs, num_times = norm_magnitude.shape
        output_image = torch.stack([norm_magnitude, norm_phase], dim=2)
        output_image = output_image.view(num_batches, 2 * num_features, num_freqs, num_times)
        return output_image


def init_stft_embedder(embedder: STFTEmbedder, full_train_signal_np: np.ndarray) -> None:
    embedder.cache_min_max_params(full_train_signal_np)

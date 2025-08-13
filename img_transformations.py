from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import pywt
import numpy as np
import torchaudio.transforms as T
from utils.utils_data import MinMaxScaler, MinMaxArgs
from ssqueezepy import cwt, icwt
from ssqueezepy import Wavelet


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


class DelayEmbedder(TsImgEmbedder):
    """
    Delay embedding transformation
    """

    def __init__(self, device: torch.device, seq_len: int, delay: int, embedding: int):
        super().__init__(device, seq_len)
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None

    def pad_to_square(self, x: torch.Tensor, mask: float = 0.0) -> torch.Tensor:
        """
        Pads the input tensor x to make it square along the last two dimensions.
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (
            0, max_side - rows, 0, max_side - cols)  # Padding format: (pad_left, pad_right, pad_top, pad_bottom)

        # Padding the last two dimensions to make them square
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded

    def unpad(self, x: torch.Tensor, original_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Removes the padding from the tensor x to get back to its original shape.
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def ts_to_img(self, signal: torch.Tensor, pad: bool = True, mask: float = 0.0) -> torch.Tensor:
        batch, length, features = signal.shape
        #  if our sequences are of different lengths, this can happen with physionet and climate datasets
        if self.seq_len != length:
            self.seq_len = length

        x_image = torch.zeros((batch, features, self.embedding, self.embedding))
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        ### SPECIAL CASE
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            # end = start + (self.embedding - 1) - missing_vals
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1

        # cache the shape of the image before padding
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image.to(self.device)[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)

        return x_image

    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        img_non_square = self.unpad(img, self.img_shape)

        batch, channels, rows, cols = img_non_square.shape

        reconstructed_x_time_series = torch.zeros((batch, channels, self.seq_len))

        for i in range(cols - 1):
            start = i * self.delay
            end = start + self.embedding
            reconstructed_x_time_series[:, :, start:end] = img_non_square[:, :, :, i]

        ### SPECIAL CASE
        start = (cols - 1) * self.delay
        end = reconstructed_x_time_series[:, :, start:].shape[-1]
        reconstructed_x_time_series[:, :, start:] = img_non_square[:, :, :end, cols - 1]
        reconstructed_x_time_series = reconstructed_x_time_series.permute(0, 2, 1)

        return reconstructed_x_time_series.cuda()


class STFTEmbedder(TsImgEmbedder):
    """
    STFT transformation
    """

    def __init__(self, device: torch.device, seq_len: int, n_fft: int, hop_length: int):
        super().__init__(device, seq_len)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_real, self.max_real, self.min_imag, self.max_imag = None, None, None, None
    
    def cache_min_max_params(self, train_data: torch.Tensor) -> None:
        """
        Args:
            train_data: training timeseries dataset. shape: B*L*K
        this function initializes the min and max values for the real and imaginary parts.
        we'll use this function only once, before the training loop starts.
        """
        real, imag = self.stft_transform(train_data)
        # compute and cache min and max values
        real, min_real, max_real = MinMaxScaler(real.numpy(), True)
        imag, min_imag, max_imag = MinMaxScaler(imag.numpy(), True)
        self.min_real, self.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
        self.min_imag, self.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)

    def stft_transform(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:    
            data: time series data. Shape: B*L*K
        Returns:
            real and imaginary parts of the STFT transformation
        """
        data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
        spec = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, center=True, power=None).to(data.device)
        transformed_data = spec(data)
        return transformed_data.real, transformed_data.imag
    
    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        assert self.min_real is not None, "use init_norm_args() to compute scaling arguments"
        # convert to complex spectrogram
        real, imag = self.stft_transform(signal)
        # MinMax scaling
        real = (MinMaxArgs(real, self.min_real.to(self.device), self.max_real.to(self.device)) - 0.5) * 2
        imag = (MinMaxArgs(imag, self.min_imag.to(self.device), self.max_imag.to(self.device)) - 0.5) * 2
        # stack real and imag parts
        stft_out = torch.cat((real, imag), dim=1)
        return stft_out

    def img_to_ts(self, x_image: torch.Tensor) -> torch.Tensor:
        n_fft = self.n_fft
        hop_length, length = self.hop_length, self.seq_len
        min_real, max_real, min_imag, max_imag = self.min_real.to(
            self.device), self.max_real.to(
            self.device), \
            self.min_imag.to(self.device), self.max_imag.to(
            self.device)
        # -- combine real and imaginary parts --
        split = torch.split(x_image, x_image.shape[1] // 2,
                            dim=1)  # x_image.shape[1] is twice the size of the original dim

        real, imag = split[0], split[1]
        unnormalized_real = ((real / 2) + 0.5) * (max_real - min_real) + min_real
        unnormalized_imag = ((imag / 2) + 0.5) * (max_imag - min_imag) + min_imag
        unnormalized_stft = torch.complex(unnormalized_real, unnormalized_imag)
        # -- inverse stft --
        ispec = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length, center=True).to(self.device)

        x_time_series = ispec(unnormalized_stft, length)

        return torch.permute(x_time_series, (0, 2, 1))  # B*L*K(C)


class WAVEmbedder(TsImgEmbedder):
    def __init__(self, device, seq_len, nv=8, scales='log-piecewise'):
        super().__init__(device,seq_len)
        
        self.wavelet = Wavelet(('gmw', {'gamma': 3, 'beta': 60, 'norm': 'energy'}))
        self.nv = nv
        self.scales = scales
        self.min_real, self.max_real = None, None
        self.min_imag, self.max_imag = None, None
        self.min_trd, self.max_trd = None, None

        # self.org_img_width = seq_len

        # self.device = device

    def cache_min_max_params(self, train_data):

        real, imag, trd = self.wav_transform(train_data)
        # compute and cache min and max values
        real, min_real, max_real = MinMaxScaler(real, True)
        imag, min_imag, max_imag = MinMaxScaler(imag, True)
        trd, min_trd, max_trd = MinMaxScaler(trd, True)

        self.min_real, self.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
        self.min_imag, self.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)
        self.min_trd, self.max_trd = torch.Tensor(min_trd), torch.Tensor(max_trd)

    def wav_transform(self, data):
        """
        Args:
            data: time series data. Shape: (B,L,K)
        Returns:
            real and imaginary part of the continuous wavelet transform 
        """

        signal = data
        t = np.arange(signal.shape[1])

        # print("signal shape", signal[1].shape)

        self.trends = np.empty_like(signal)
        # self.trend_2D = np.meshgrid
        self.detrended_signals = np.empty_like(signal)
        self.trend_2D = np.empty((signal.shape[0], signal.shape[1], signal.shape[1]))

        for i in range(signal.shape[0]):
            trend = np.poly1d(np.polyfit(t, signal[i].squeeze(-1), deg=7))(t)
            trend = trend[:,np.newaxis]
            # print(trend.shape)
            self.trends[i] = trend
            self.detrended_signals[i] = signal[i] - trend

            T, Z = np.meshgrid(t,t)
            trend_2D = np.poly1d(np.polyfit(t, signal[i].squeeze(-1), deg=7))(T)
            self.trend_2D[i] = trend_2D

        self.detrended_signals = self.detrended_signals.squeeze(-1)
        # print("testhh",self.detrended_signals.shape)

        Wx_detrended, self.scales_s = cwt(self.detrended_signals, wavelet=self.wavelet, scales = self.scales,nv=self.nv,l1_norm=False)

        Wx_detrended = Wx_detrended[:,np.newaxis,:,:] #add newaxis on dim1 (21, 1, 227, 728)
        self.trend_2D = self.trend_2D[:,np.newaxis,:,:]
        # print("trend",self.trends.shape)
        # print("trend_2D",self.trend_2D.shape)

        return Wx_detrended.real, Wx_detrended.imag, self.trend_2D

    def ts_to_img(self, signal):
        assert self.min_real is not None, "use init_norm_args() to compute scaling arguments"

        real, imag, trd = self.wav_transform(signal)
        real, imag, trd = torch.Tensor(real).to(self.device), torch.Tensor(imag).to(self.device), torch.Tensor(trd).to(self.device)
        
        # MinMax scaling
        real = (MinMaxArgs(real, self.min_real.to(self.device), self.max_real.to(self.device)) - 0.5) * 2
        imag = (MinMaxArgs(imag, self.min_imag.to(self.device), self.max_imag.to(self.device)) - 0.5) * 2
        trd = (MinMaxArgs(trd, self.min_trd.to(self.device), self.max_trd.to(self.device)) - 0.5) * 2

        #padding on image
        real = self.pad_to_square_wav(real)
        imag = self.pad_to_square_wav(imag)

        wavelet_out = torch.cat((real, imag, trd), dim=1)
        # print("wavelet_out",wavelet_out.shape)

        return wavelet_out

    def img_to_ts(self, x_image):
        
        min_real, max_real = self.min_real.to(self.device), self.max_real.to(self.device)
        min_imag, max_imag = self.min_imag.to(self.device), self.max_imag.to(self.device)
        min_trd, max_trd = self.min_trd.to(self.device), self.max_trd.to(self.device)

        split = torch.split(x_image, x_image.shape[1] // 3, dim=1)
        real, imag, trd = split[0], split[1], split[2]
        
        #unpad of image
        real = self.unpad_wav(real)
        imag = self.unpad_wav(imag)

        unnormalized_real = ((real / 2) + 0.5) * (max_real - min_real) + min_real
        unnormalized_imag = ((imag / 2) + 0.5) * (max_imag - min_imag) + min_imag
        unnormalized_trd  = ((trd / 2)  + 0.5) * (max_trd  - min_trd) + min_trd

        unnormalized_wav = torch.complex(unnormalized_real, unnormalized_imag)

        unnormalized_wav = unnormalized_wav.cpu().numpy()
        unnormalized_trd = unnormalized_trd.cpu().numpy()

        # print("wave",unnormalized_wav.shape)
        # print("tred",unnormalized_trd.shape)

        Wx_detrended = np.squeeze(unnormalized_wav,axis=1) #remove the added axis
        trd_img = np.squeeze(unnormalized_trd,axis=1) #remove the added axis

        reconstructed_detrended = icwt(Wx_detrended, wavelet=self.wavelet, scales = self.scales_s,nv=self.nv,l1_norm=False)

        self.reconstructed_trend = np.empty((trd_img.shape[0],trd_img.shape[1]))

        for i in range(trd_img.shape[0]):
            t = np.arange(trd_img.shape[1])
            trend_values = trd_img[i][0, :]  # Extract the corresponding 1D trend values
            reconstructed_polynomial = np.poly1d(np.polyfit(t, trend_values, deg=7))
            self.reconstructed_trend[i] = reconstructed_polynomial(t)

        # print("final_reconstructed_detrend",reconstructed_detrended.shape)
        # print("final_reconstructed_trend",self.reconstructed_trend.shape)

        reconstructed_with_trend = reconstructed_detrended + self.reconstructed_trend

        return torch.Tensor(reconstructed_with_trend).to(self.device)

    def pad_to_square_wav(self, x_image):
        _, _, height, width = x_image.shape
        height_padding = width - height 
        
        if height_padding > 0:
            # print("pad")
            last_row = x_image[:, :, -1:, :] 
            row_padding_tensor = last_row.repeat(1, 1, height_padding, 1)  
            x_padded_image = torch.cat((x_image, row_padding_tensor), dim=2)  

        else: 
            # print("nopad")
            return x_image

        return x_padded_image

    def unpad_wav(self, x_image):

        org_img_width = self.scales_s.shape[0] #orginal number of scale

        _, _, img_height, img_width = x_image.shape

        if img_width != org_img_width:

            return x_image[: , : , :org_img_width, :]

        return x_image


class WAVEmbedder_ST(TsImgEmbedder):
    def __init__(self, device, seq_len, nv=7, scales='log-piecewise'):
        super().__init__(device,seq_len)
        self.wavelet = Wavelet(('gmw', {'gamma': 3, 'beta': 60, 'norm': 'energy'}))
        self.nv = nv
        self.scales = scales
        self.min_real, self.max_real = None, None
        self.min_imag, self.max_imag = None, None
        self.min_trd, self.max_trd = None, None

    def cache_min_max_params(self, train_data):
        real, imag, trd = self.wav_transform(train_data)
        real, min_real, max_real = MinMaxScaler(real, True)
        imag, min_imag, max_imag = MinMaxScaler(imag, True)
        trd, min_trd, max_trd = MinMaxScaler(trd, True)

        self.min_real, self.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
        self.min_imag, self.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)
        self.min_trd, self.max_trd = torch.Tensor(min_trd), torch.Tensor(max_trd)

    def wav_transform(self, data):
        signal = data
        t = np.arange(signal.shape[1])

        self.trends = np.empty_like(signal)
        self.detrended_signals = np.empty_like(signal)
        self.trends_2D = np.empty((signal.shape[0], signal.shape[2], signal.shape[1], signal.shape[1]))

        for batch_idx in range(signal.shape[0]):
            for feature_idx in range(signal.shape[2]):
                time_series_signal = signal[batch_idx,:,feature_idx]
                trend = np.poly1d(np.polyfit(t, time_series_signal, deg=7))(t)
                self.trends[batch_idx, :, feature_idx] = trend
                detrended_signal = time_series_signal - trend
                self.detrended_signals[batch_idx,:,feature_idx] = detrended_signal

                T, Z = np.meshgrid(t,t)
                trend_2D = np.poly1d(np.polyfit(t, time_series_signal, deg=7))(T)
                self.trends_2D[batch_idx,feature_idx,:,:] = trend_2D

        wavelogram_real = []
        wavelogram_imag = []

        for feature_wav_idx in range(signal.shape[2]):
            ts_to_wav_signal = self.detrended_signals[:,:,feature_wav_idx]
            Wx_detrended, self.scales_s = cwt(ts_to_wav_signal, wavelet=self.wavelet, scales = self.scales,nv=self.nv,l1_norm=False)
            wavelogram_real.append(Wx_detrended.real)
            wavelogram_imag.append(Wx_detrended.imag)

        wavelogram_real = np.array(wavelogram_real)
        wavelogram_imag = np.array(wavelogram_imag)
        wavelogram_real = np.transpose(wavelogram_real, (1, 0, 2, 3))
        wavelogram_imag = np.transpose(wavelogram_imag, (1, 0, 2, 3))

        return wavelogram_real, wavelogram_imag, self.trends_2D

    def ts_to_img(self, signal):
        assert self.min_real is not None, "use init_norm_args() to compute scaling arguments"

        # Signal is expected to be (B, L, K) for wav_transform
        # Here, B will be 1 as we are passing the whole signal at once.
        real, imag, trd = self.wav_transform(signal.cpu().numpy())
        real, imag, trd = torch.Tensor(real).to(self.device), torch.Tensor(imag).to(self.device), torch.Tensor(trd).to(self.device)

        # MinMax scaling
        real = (MinMaxArgs(real, self.min_real.to(self.device), self.max_real.to(self.device)) - 0.5) * 2
        imag = (MinMaxArgs(imag, self.min_imag.to(self.device), self.max_imag.to(self.device)) - 0.5) * 2
        trd = (MinMaxArgs(trd, self.min_trd.to(self.device), self.max_trd.to(self.device)) - 0.5) * 2

        real = self.pad_to_square_wav(real)
        imag = self.pad_to_square_wav(imag)

        wavelet_out = torch.cat((real, imag, trd), dim=1)
        return wavelet_out

    def img_to_ts(self, x_image):
        min_real, max_real = self.min_real.to(self.device), self.max_real.to(self.device)
        min_imag, max_imag = self.min_imag.to(self.device), self.max_imag.to(self.device)
        min_trd, max_trd = self.min_trd.to(self.device), self.max_trd.to(self.device)

        split = torch.split(x_image, x_image.shape[1] // 3, dim=1)
        real, imag, trd = split[0], split[1], split[2]

        real = self.unpad_wav(real)
        imag = self.unpad_wav(imag)

        unnormalized_real = ((real / 2) + 0.5) * (max_real - min_real) + min_real
        unnormalized_imag = ((imag / 2) + 0.5) * (max_imag - min_imag) + min_imag
        unnormalized_trd  = ((trd / 2)  + 0.5) * (max_trd  - min_trd) + min_trd

        unnormalized_wav = torch.complex(unnormalized_real, unnormalized_imag)

        unnormalized_wav = unnormalized_wav.cpu().numpy()
        unnormalized_trd = unnormalized_trd.cpu().numpy()

        self.reconstructed_detrended = np.empty((unnormalized_wav.shape[0],unnormalized_wav.shape[3],unnormalized_wav.shape[1]))

        for ft_iwav_idx in range(unnormalized_wav.shape[1]):
            iwx_detrended = unnormalized_wav[:,ft_iwav_idx,:,:]
            reconstructed_detrendeds = icwt(iwx_detrended, wavelet=self.wavelet, scales = self.scales,nv=self.nv,l1_norm=False)
            self.reconstructed_detrended[:,:,ft_iwav_idx] = reconstructed_detrendeds

        self.reconstructed_trend = np.empty((unnormalized_trd.shape[0],unnormalized_trd.shape[3],unnormalized_trd.shape[1]))

        for bt_trd_idx in range(unnormalized_trd.shape[0]):
            for ft_trx_idx in range(unnormalized_trd.shape[1]):
                t = np.arange(unnormalized_trd.shape[3])
                trend_values = unnormalized_trd[bt_trd_idx][ft_trx_idx][0, :]
                reconstructed_polynomial = np.poly1d(np.polyfit(t, trend_values, deg=7))
                self.reconstructed_trend[bt_trd_idx,:,ft_trx_idx] = reconstructed_polynomial(t)

        reconstructed_with_trend = self.reconstructed_detrended + self.reconstructed_trend

        return torch.Tensor(reconstructed_with_trend).to(self.device)

    def pad_to_square_wav(self, x_image):
        _, _, height, width = x_image.shape
        height_padding = width - height

        if height_padding > 0:
            last_row = x_image[:, :, -1:, :]
            row_padding_tensor = last_row.repeat(1, 1, height_padding, 1)
            x_padded_image = torch.cat((x_image, row_padding_tensor), dim=2)
        else:
            return x_image

        return x_padded_image

    def unpad_wav(self, x_image):
        # org_img_width = self.scales_s.shape[0] # The original number of scales is needed here
        # self.scales_s is set in wav_transform. When unpadding, we need this value.
        # This becomes tricky if wav_transform hasn't been called yet for the full signal,
        # or if it was called for a different length signal.
        # A more robust way might be to pass the original_scales_s or calculate it.
        # For simplicity here, let's assume it's set after the full signal transform.
        org_img_width = self.scales_s.shape[0] # Assuming self.scales_s is set from full transform

        _, _, img_height, img_width = x_image.shape

        if img_width != org_img_width:
            return x_image[: , : , :org_img_width, :]
        return x_image
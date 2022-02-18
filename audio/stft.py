import torch
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
import librosa
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn
import pyloudnorm as pyln

from audio.audio_processing import (
    dynamic_range_compression,
    dynamic_range_decompression,
    window_sumsquare,
)
from audio.tools import (
    librosa_pad_lr,
    amp_to_db,
    normalize,
)


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data.cuda(),
            torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda(),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        energy = torch.norm(magnitudes, dim=1)

        return mel_output, energy


class FastSpeechSTFT(torch.nn.Module):
    def __init__(
        self,
        fft_size,
        hop_size,
        win_length,
        num_mels,
        sample_rate,
        fmin,
        fmax,
        window='hann',
        eps=1e-10,
        loud_norm=False,
        min_level_db=-100,
    ):
        super(FastSpeechSTFT, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.window = window
        self.eps = eps
        self.loud_norm = loud_norm
        self.min_level_db = min_level_db

    def mel_spectrogram(self, wav, return_linear=False):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        wav: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        if self.loud_norm:
            meter = pyln.Meter(self.sample_rate)  # create BS.1770 meter
            loudness = meter.integrated_loudness(wav)
            wav = pyln.normalize.loudness(wav, loudness, -22.0)
            if np.abs(wav).max() > 1:
                wav = wav / np.abs(wav).max()

        # get amplitude spectrogram
        x_stft = librosa.stft(wav, n_fft=self.fft_size, hop_length=self.hop_size,
                            win_length=self.win_length, window=self.window, pad_mode="constant")
        spc = np.abs(x_stft)  # (n_bins, T)

        # get mel basis
        fmin = 0 if self.fmin == -1 else self.fmin
        fmax = sample_rate / 2 if self.fmax == -1 else self.fmax
        mel_basis = librosa.filters.mel(self.sample_rate, self.fft_size, self.num_mels, self.fmin, self.fmax)
        mel = mel_basis @ spc

        # get log scaled mel
        mel = np.log10(np.maximum(self.eps, mel))

        l_pad, r_pad = librosa_pad_lr(wav, self.fft_size, self.hop_size, 1)
        wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
        wav = wav[:mel.shape[1] * self.hop_size]

        # get energy
        energy = np.sqrt(np.exp(mel) ** 2).sum(-1)

        if not return_linear:
            return wav, mel, energy
        else:
            spc = amp_to_db(spc)
            spc = normalize(spc, self.min_level_db)
            return wav, mel, energy, spc

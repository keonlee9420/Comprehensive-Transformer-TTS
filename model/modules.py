import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from numba import jit, prange
import numpy as np
import torch.nn.functional as F

from utils.tools import (
    get_variance_level,
    get_phoneme_level_pitch,
    get_phoneme_level_energy,
    get_mask_from_lengths,
    pad_1D,
    pad,
)
from .transformers.blocks import LinearNorm, ConvNorm, ConvBlock, ConvBlock2D


@jit(nopython=True)
def mas_width1(attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]): # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x


class ProsodyExtractor(nn.Module):
    """ Prosody Extractor """

    def __init__(self, n_mel_channels, d_model, kernel_size):
        super(ProsodyExtractor, self).__init__()
        self.d_model = d_model
        self.conv_stack = nn.Sequential(
            ConvBlock2D(
                in_channels=1,
                out_channels=self.d_model,
                kernel_size=kernel_size,
            ),
            ConvBlock2D(
                in_channels=self.d_model,
                out_channels=1,
                kernel_size=kernel_size,
            ),
        )
        self.gru = nn.GRU(
            input_size=n_mel_channels,
            hidden_size=self.d_model,
            batch_first=True,
            bidirectional=True,
        )

    def get_prosody_embedding(self, mel):
        """
        mel -- [B, mel_len, n_mel_channels], B=1
        h_n -- [B, 2 * d_model], B=1
        """
        x = self.conv_stack(mel.unsqueeze(-1)).squeeze(-1)
        _, h_n = self.gru(x)
        h_n = torch.cat((h_n[0], h_n[1]), dim=-1)
        return h_n

    def forward(self, mel, mel_len, duration, src_len):
        """
        mel -- [B, mel_len, n_mel_channels]
        mel_len -- [B,]
        duration -- [B, src_len]
        src_len -- [B,]
        batch -- [B, src_len, 2 * d_model]
        """
        batch = []
        for m, m_l, d, s_l in zip(mel, mel_len, duration, src_len):
            b = []
            for m_p in torch.split(m[:m_l], list(d[:s_l].int()), dim=0):
                b.append(self.get_prosody_embedding(m_p.unsqueeze(0)).squeeze(0))
            batch.append(torch.stack(b, dim=0))

        return pad(batch)


class MDN(nn.Module):
    """ Mixture Density Network """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.w = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=-1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, x):
        """
        x -- [B, src_len, in_features]
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        """
        B, src_len, _ = x.shape
        w = self.w(x)
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(B, src_len, self.num_gaussians, self.out_features)
        mu = self.mu(x)
        mu = mu.view(B, src_len, self.num_gaussians, self.out_features)
        return w, sigma, mu


class ProsodyPredictor(nn.Module):
    """ Prosody Predictor """

    def __init__(self, d_model, kernel_size, num_gaussians, dropout):
        super(ProsodyPredictor, self).__init__()
        self.d_model = d_model
        self.conv_stack = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.d_model,
                    out_channels=self.d_model,
                    kernel_size=kernel_size[i],
                    dropout=dropout,
                    normalization=nn.LayerNorm,
                    transpose=True,
                )
                for i in range(2)
            ]
        )
        self.gru_cell = nn.GRUCell(
            self.d_model + 2 * self.d_model,
            2 * self.d_model,
        )
        self.gmm_mdn = MDN(
            in_features=2 * self.d_model,
            out_features=2 * self.d_model,
            num_gaussians=num_gaussians,
        )

    def init_state(self, x):
        """
        x -- [B, src_len, d_model]
        p_0 -- [B, 2 * d_model]
        self.gru_hidden -- [B, 2 * d_model]
        """
        B, _, d_model = x.shape
        p_0 = torch.zeros((B, 2 * d_model), device=x.device, requires_grad=True)
        self.gru_hidden = torch.zeros((B, 2 * d_model), device=x.device, requires_grad=True)
        return p_0

    def forward(self, h_text, mask=None):
        """
        h_text -- [B, src_len, d_model]
        mask -- [B, src_len]
        outputs -- [B, src_len, 2 * d_model]
        """
        x = h_text
        for conv_layer in self.conv_stack:
            x = conv_layer(x, mask=mask)

        # Autoregressive Prediction
        p_0 = self.init_state(x)

        outputs = [p_0]
        for i in range(x.shape[1]):
            p_input = torch.cat((x[:, i], outputs[-1]), dim=-1) # [B, 3 * d_model]
            self.gru_hidden = self.gru_cell(p_input, self.gru_hidden) # [B, 2 * d_model]
            outputs.append(self.gru_hidden)
        outputs = torch.stack(outputs[1:], dim=1) # [B, src_len, 2 * d_model]

        # GMM-MDN
        w, sigma, mu = self.gmm_mdn(outputs)
        if mask is not None:
            w = w.masked_fill(mask.unsqueeze(-1), 0 if self.training else 1e-9) # 1e-9 for categorical sampling
            sigma = sigma.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
            mu = mu.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)

        return w, sigma, mu

    @staticmethod
    def sample(w, sigma, mu, mask=None):
        """ Draw samples from a GMM-MDN 
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        mask -- [B, src_len]
        output -- [B, src_len, out_features]
        """
        from torch.distributions import Categorical
        batch = []
        for i in range(w.shape[1]):
            w_i, sigma_i, mu_i = w[:, i], sigma[:, i], mu[:, i]
            ws = Categorical(w_i).sample().view(w_i.size(0), 1, 1)
            # Choose a random sample, one randn for batch X output dims
            # Do a (output dims)X(batch size) tensor here, so the broadcast works in
            # the next step, but we have to transpose back.
            gaussian_noise = torch.randn(
                (sigma_i.size(2), sigma_i.size(0)), requires_grad=False).to(w.device)
            variance_samples = sigma_i.gather(1, ws).detach().squeeze()
            mean_samples = mu_i.detach().gather(1, ws).squeeze()
            batch.append((gaussian_noise * variance_samples + mean_samples).transpose(0, 1))
        output = torch.stack(batch, dim=1)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config, train_config, d_model):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config, d_model)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config, d_model)
        self.energy_predictor = VariancePredictor(model_config, d_model)

        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.binarization_start_steps = train_config["duration"]["binarization_start_steps"]
        if model_config["duration_modeling"]["learn_alignment"]:
            self.aligner = AlignmentEncoder(
                n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                n_att_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                n_text_channels=d_model,
                temperature=model_config["duration_modeling"]["aligner_temperature"],
                multi_speaker=model_config["multi_speaker"],
            )

        self.learn_prosody = model_config["learn_prosody"]
        self.learn_mixture = model_config["prosody"]["learn_mixture"]
        if self.learn_prosody:
            if self.learn_mixture:
                assert not self.learn_alignment
                self.prosody_extractor = ProsodyExtractor(
                    n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                    d_model=d_model,
                    kernel_size=model_config["prosody"]["extractor_kernel_size"],
                )
                self.prosody_predictor = ProsodyPredictor(
                    d_model=d_model,
                    kernel_size=model_config["prosody"]["predictor_kernel_size"],
                    num_gaussians=model_config["prosody"]["predictor_num_gaussians"],
                    dropout=model_config["prosody"]["predictor_dropout"],
                )
                self.prosody_linear = LinearNorm(2 * d_model, d_model)

        pitch_level_tag, energy_level_tag, self.pitch_feature_level, self.energy_feature_level = \
                                    get_variance_level(preprocess_config, model_config, data_loading=False)

        # Note that there is no pre-extracted phoneme-level variance features in unsupervised duration modeling.
        # Alternatively, we can use convolutional embedding instead of bucket-based embedding in such cases.
        self.use_conv_embedding = self.learn_alignment \
            and (self.pitch_feature_level == "phoneme_level" or self.energy_feature_level == "phoneme_level")
        if self.use_conv_embedding:
            kernel_size = model_config["variance_embedding"]["kernel_size"]
            self.pitch_embedding = ConvNorm(
                1, 
                d_model,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                bias=False,
                w_init_gain="tanh",
                transpose=True,
            )
            self.energy_embedding = ConvNorm(
                1,
                d_model,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                bias=False,
                w_init_gain="tanh",
                transpose=True,
            )
        else:
            pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
            energy_quantization = model_config["variance_embedding"]["energy_quantization"]
            n_bins = model_config["variance_embedding"]["n_bins"]
            assert pitch_quantization in ["linear", "log"]
            assert energy_quantization in ["linear", "log"]
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
            ) as f:
                stats = json.load(f)
                pitch_min, pitch_max = stats[f"pitch_{pitch_level_tag}"][:2]
                energy_min, energy_max = stats[f"energy_{energy_level_tag}"][:2]

            if pitch_quantization == "log":
                self.pitch_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.pitch_bins = nn.Parameter(
                    torch.linspace(pitch_min, pitch_max, n_bins - 1),
                    requires_grad=False,
                )
            if energy_quantization == "log":
                self.energy_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.energy_bins = nn.Parameter(
                    torch.linspace(energy_min, energy_max, n_bins - 1),
                    requires_grad=False,
                )

            self.pitch_embedding = nn.Embedding(
                n_bins, d_model
            )
            self.energy_embedding = nn.Embedding(
                n_bins, d_model
            )

    def binarize_attention_parallel(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS.
        These will no longer recieve a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
        return torch.from_numpy(attn_out).to(attn.device)

    def get_phoneme_level_pitch(self, duration, src_len, pitch_frame):
        return torch.from_numpy(
            pad_1D(
                [get_phoneme_level_pitch(dur[:len], var) for dur, len, var \
                        in zip(duration.int().cpu().numpy(), src_len.cpu().numpy(), pitch_frame.cpu().numpy())]
            )
        ).float().to(pitch_frame.device)

    def get_phoneme_level_energy(self, duration, src_len, energy_frame):
        return torch.from_numpy(
            pad_1D(
                [get_phoneme_level_energy(dur[:len], var) for dur, len, var \
                        in zip(duration.int().cpu().numpy(), src_len.cpu().numpy(), energy_frame.cpu().numpy())]
            )
        ).float().to(energy_frame.device)

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(target.unsqueeze(-1)) if self.use_conv_embedding \
                else self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(prediction.unsqueeze(-1)) if self.use_conv_embedding \
                else self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding =  self.energy_embedding(target.unsqueeze(-1)) if self.use_conv_embedding \
                else self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(prediction.unsqueeze(-1)) if self.use_conv_embedding \
                else self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        speaker_embedding,
        text,
        text_embedding,
        src_len,
        src_mask,
        mel,
        mel_len,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        attn_prior=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
    ):
        x = text
        if speaker_embedding is not None:
            x = x + speaker_embedding.unsqueeze(1).expand(
                -1, text.shape[1], -1
            )

        prosody_info = None
        if self.learn_prosody:
            # GMM-MDN for Phone-Level Prosody Modelling
            if self.learn_mixture and not self.learn_alignment:
                w, sigma, mu = self.prosody_predictor(text, src_mask)

                if self.training:
                    prosody_embeddings = self.prosody_extractor(mel, mel_len, duration_target, src_len)
                else:
                    prosody_embeddings = self.prosody_predictor.sample(w, sigma, mu)
                x = x + self.prosody_linear(prosody_embeddings)
                prosody_info = (w, sigma, mu, prosody_embeddings)

        log_duration_prediction = self.duration_predictor(x, src_mask)
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            min=0,
        )

        # Trainig of unsupervised duration modeling
        attn_soft, attn_hard, attn_hard_dur, attn_logprob = None, None, None, None
        if attn_prior is not None:
            assert self.learn_alignment and duration_target is None and mel is not None
            attn_soft, attn_logprob = self.aligner(
                mel.transpose(1, 2),
                text_embedding.transpose(1, 2),
                src_mask.unsqueeze(-1),
                attn_prior.transpose(1, 2),
                speaker_embedding,
            )
            attn_hard = self.binarize_attention_parallel(attn_soft, src_len, mel_len)
            attn_hard_dur = attn_hard.sum(2)[:, 0, :]
        attn_out = (attn_soft, attn_hard, attn_hard_dur, attn_logprob)

        # Note that there is no pre-extracted phoneme-level variance features in unsupervised duration modeling.
        # Alternatively, we can use attn_hard_dur instead of duration_target for computing phoneme-level variances.
        if self.pitch_feature_level == "phoneme_level":
            if attn_prior is not None:
                pitch_target = self.get_phoneme_level_pitch(attn_hard_dur, src_len, pitch_target)
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, src_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            if attn_prior is not None:
                energy_target = self.get_phoneme_level_energy(attn_hard_dur, src_len, energy_target)
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, src_mask, e_control)
            x = x + energy_embedding

        # Upsampling from src length to mel length
        if attn_prior is not None: # Trainig of unsupervised duration modeling
            if step < self.binarization_start_steps:
                A_soft = attn_soft.squeeze(1)
                x = torch.bmm(A_soft,x)
            else:
                x, mel_len = self.length_regulator(x, attn_hard_dur, max_len)
            duration_rounded = attn_hard_dur
        elif duration_target is not None: # Trainig of supervised duration modeling
            assert not self.learn_alignment and attn_prior is None
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else: # Inference
            assert attn_prior is None and duration_target is None
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(x, pitch_target, mel_mask, p_control)
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(x, energy_target, mel_mask, e_control)
            x = x + energy_embedding

        return (
            x,
            pitch_target,
            pitch_prediction,
            energy_target,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
            attn_out,
            prosody_info,
        )


class AlignmentEncoder(torch.nn.Module):
    """ Alignment Encoder for Unsupervised Duration Modeling """

    def __init__(self, 
                n_mel_channels,
                n_att_channels,
                n_text_channels,
                temperature,
                multi_speaker):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu'
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_text_channels * 2,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                n_mel_channels,
                n_mel_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu',
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_mel_channels * 2,
                n_mel_channels,
                kernel_size=1,
                bias=True,
            ),
            torch.nn.ReLU(),
            ConvNorm(
                n_mel_channels,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        if multi_speaker:
            self.key_spk_proj = LinearNorm(n_text_channels, n_text_channels)
            self.query_spk_proj = LinearNorm(n_text_channels, n_mel_channels)

    def forward(self, queries, keys, mask=None, attn_prior=None, speaker_embed=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
            speaker_embed (torch.tensor): B x C tnesor of speaker embedding for multi-speaker scheme.
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        if speaker_embed is not None:
            keys = keys + self.key_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, keys.shape[-1], -1
            )).transpose(1, 2)
            queries = queries + self.query_spk_proj(speaker_embed.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            #print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)
            #print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config, input_size):
        super(VariancePredictor, self).__init__()

        self.input_size = input_size
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        ConvNorm(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=(self.kernel - 1) // 2,
                            dilation=1,
                            transpose=True,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        ConvNorm(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            stride=1,
                            padding=1,
                            dilation=1,
                            transpose=True,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

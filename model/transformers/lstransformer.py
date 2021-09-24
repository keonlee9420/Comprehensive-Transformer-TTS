import functools
from math import gcd, ceil
import torch
from torch import nn, einsum
import numpy as np
from torch.nn import functional as F
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from einops import rearrange, repeat

from text.symbols import symbols

from .constants import PAD
from .blocks import (
    get_sinusoid_encoding_table,
    LinearNorm,
)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def lcm(*numbers):
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))


def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple

    if m.is_integer():
        return tensor

    remainder = ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)


class TextEncoder(nn.Module):
    """ Text Encoder """

    def __init__(self, config):
        super(TextEncoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_head = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = FFTBlock(
            n_layers, d_model, n_head, d_head, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, src_seq, mask):

        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Forward
        src_word_emb = self.src_word_emb(src_seq)
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = src_word_emb + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = src_word_emb + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        enc_output = self.layer_stack(enc_output, mask=mask)

        return enc_output, src_word_emb


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_head = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = FFTBlock(
            n_layers, d_model, n_head, d_head, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_seq, mask):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]

        dec_output = self.layer_stack(dec_output, mask=mask)

        return dec_output, mask


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, depth, d_model, n_head, d_head, d_inner, kernel_size, dropout=0.1, causal=True, segment_size=16, r=1):
        super(FFTBlock, self).__init__()

        segment_size = default(segment_size, 16 if causal else None)
        r = default(r, 1 if causal else 128)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            attn = LongShortAttention(d_model, d_head, n_head, segment_size=segment_size, r=r, dropout=dropout)
            ff = PositionwiseFeedForward(
                d_model, d_inner, kernel_size, dropout=dropout
            )
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, attn),
                PreNorm(d_model, ff)
            ]))

    def forward(self, x, mask=None):

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = x.masked_fill(mask.unsqueeze(-1), 0)

            x = ff(x) + x
            x = x.masked_fill(mask.unsqueeze(-1), 0)

        return x
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class LongShortAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = True,
        window_size = 128,
        pos_emb = None,
        segment_size = 16,
        r = 1,
        dropout = 0.
    ):
        super().__init__()
        assert not (causal and r >= segment_size), 'r should be less than segment size, if autoregressive'

        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal

        self.window_size = window_size
        self.segment_size = segment_size
        self.pad_to_multiple = window_size if not causal else lcm(window_size, segment_size)

        self.to_dynamic_proj = nn.Linear(dim_head, r, bias = False)
        self.local_norm = nn.LayerNorm(dim_head)
        self.global_norm = nn.LayerNorm(dim_head)

        self.pos_emb = default(pos_emb, RotaryEmbedding(dim_head))

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        b, n, *_, h, device, causal, w, s = *x.shape, self.heads, x.device, self.causal, self.window_size, self.segment_size

        # pad input sequence to multiples of window size (or window size and segment length if causal)

        x = pad_to_multiple(x, self.pad_to_multiple, dim = -2, value = 0.)

        # derive from variables

        padded_len = x.shape[-2]
        windows = padded_len // w
        is_padded = padded_len != n

        mask_value = -torch.finfo(x.dtype).max

        # handle mask if padding was needed and mask was not given

        if is_padded:
            mask = default(mask, torch.ones((b, n), device = device).bool())
            mask = pad_to_multiple(mask, w, dim = -1, value = False)

        # get queries, keys, values

        qkv = (self.to_q(x), self.to_kv(x))

        # get sequence range, for calculating mask

        seq_range = torch.arange(padded_len, device = device)

        # split heads

        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        # rotary embedding

        if exists(self.pos_emb):
            rotary_emb = self.pos_emb(seq_range, cache_key = padded_len)
            rotary_emb = rearrange(rotary_emb, 'n d -> () n d')
            q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        # scale queries

        q = q * self.scale

        # get local queries and keys similarity scores

        window_fn = lambda t: rearrange(t, 'b (w n) d -> b w n d', n = w)
        lq, lkv = map(window_fn, (q, kv))

        lookaround_kwargs = {'backward': 1, 'forward': (0 if causal else 1)}
        lkv = look_around(lkv, **lookaround_kwargs)

        lkv = self.local_norm(lkv)
        lsim = einsum('b w i d, b w j d -> b w i j', lq, lkv)

        # prepare global key / values

        if self.causal:
            # autoregressive global attention is handled in segments
            # later on, these segments are carefully masked to prevent leakage

            gkv = rearrange(kv, 'b (n s) d -> b n s d', s = s)
            pkv = self.to_dynamic_proj(gkv)

            if exists(mask):
                pmask = repeat(mask, 'b (n s) -> (b h) n s', s = s, h = h)
                pkv.masked_fill_(~pmask[..., None], mask_value)

            pkv = pkv.softmax(dim = -2)

            gkv = einsum('b n s d, b n s r -> b n r d', gkv, pkv)
            gkv = rearrange(gkv, 'b n r d -> b (n r) d')
        else:
            # equation (3) in the paper

            pkv = self.to_dynamic_proj(kv)

            if exists(mask):
                pkv.masked_fill_(~mask[..., None], mask_value)

            pkv = pkv.softmax(dim = -2)

            gkv = einsum('b n d, b n r -> b r d', kv, pkv)

        # calculate global queries and keys similarity scores

        gkv = self.global_norm(gkv)
        gsim = einsum('b n d, b r d -> b n r', q, gkv)

        # concat values together (same as keys)

        gkv = repeat(gkv, 'b r d -> b w r d', w = windows)
        v = torch.cat((gkv, lkv), dim = -2)

        # masking

        buckets, i, j = lsim.shape[-3:]

        if exists(mask):
            mask = repeat(mask, 'b (w n) -> (b h) w n', n = w, h = h)
            mask = look_around(mask, pad_value = False, **lookaround_kwargs)
            mask = rearrange(mask, 'b w n -> b w () n')
            lsim.masked_fill_(~mask, mask_value)

        # mask out padding

        seq_range_windowed = rearrange(seq_range, '(w n) -> () w n', w = windows)
        pad_mask = look_around(seq_range_windowed, pad_value = -1, **lookaround_kwargs) == -1
        lsim.masked_fill_(pad_mask[:, :, None], mask_value)

        # calculate causal masking for both global and local

        if self.causal:
            g_range = rearrange(seq_range, '(n s) -> n s', s = s)
            g_range_max = g_range.amax(dim = -1)
            g_mask = seq_range[:, None] >= g_range_max[None, :]
            g_mask = rearrange(g_mask, 'i j -> () i j')
            gsim.masked_fill_(~g_mask, mask_value)

            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            causal_mask = repeat(causal_mask, 'i j -> () u i j', u = buckets)
            lsim.masked_fill_(causal_mask, mask_value)

        # concat local and global similarities together to ready for attention

        gsim = rearrange(gsim, 'b (w n) r -> b w n r', w = windows)
        sim = torch.cat((gsim, lsim), dim = -1)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values (same as keys, since tied) and project out

        out = einsum('b w i j, b w j d -> b w i d', attn, v)
        out = rearrange(out, '(b h) w n d -> b (w n) (h d)', h = h)
        out = out[:, :n]
        return self.to_out(out)


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
        )
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w_2(F.gelu(self.w_1(output)))
        output = output.transpose(1, 2)
        return self.dropout(output)

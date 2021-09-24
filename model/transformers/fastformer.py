import torch
from torch import nn, einsum
from einops import rearrange
import numpy as np
from torch.nn import functional as F

from text.symbols import symbols

from .constants import PAD
from .blocks import (
    get_sinusoid_encoding_table,
    LinearNorm,
)


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

    def __init__(self, depth, d_model, n_head, d_head, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            attn = FastAttention(d_model, d_head, n_head, dropout=dropout)
            ff = PositionwiseFeedForward(
                d_model, d_inner, kernel_size, dropout=dropout
            )
            self.layers.append(nn.ModuleList([
                PreNorm(d_model, attn),
                PreNorm(d_model, ff)
            ]))

        # weight tie projections across all layers
        first_block, _ = self.layers[0]
        for block, _ in self.layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

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


# class FastAttention(nn.Module):
#     """ lucidrains' Fastformer Attention module """

#     def __init__(self, dim, dim_head, heads, dropout=0.1):
#         super().__init__()

#         inner_dim = heads * dim_head
#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.to_qkv = LinearNorm(dim, inner_dim * 3, bias = False)

#         self.to_q_attn_logits = LinearNorm(dim_head, 1, bias = False)  # for projecting queries to query attention logits
#         self.to_k_attn_logits = LinearNorm(dim_head, 1, bias = False)  # for projecting keys to key attention logits

#         self.to_r = LinearNorm(dim_head, dim_head)

#         self.to_out = LinearNorm(inner_dim, dim)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, mask=None):
#         h = self.heads

#         residual = x

#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
#         mask_value = -torch.finfo(q.dtype).max
#         mask = rearrange(mask, 'b n -> b () n')

#         # calculate query attention logits

#         q_attn_logits = rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
#         q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
#         q_attn = q_attn_logits.softmax(dim = -1)

#         # calculate global query token

#         global_q = einsum('b h n, b h n d -> b h d', q_attn, q)
#         global_q = rearrange(global_q, 'b h d -> b h () d')

#         # bias keys with global query token

#         k = k * global_q

#         # now calculate key attention logits

#         k_attn_logits = rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
#         k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
#         k_attn = k_attn_logits.softmax(dim = -1)

#         # calculate global key token

#         global_k = einsum('b h n, b h n d -> b h d', k_attn, k)
#         global_k = rearrange(global_k, 'b h d -> b h () d')

#         # bias the values

#         v = v * global_k
#         r = self.to_r(v)

#         r = r + q # paper says to add the queries as a residual

#         # aggregate

#         r = rearrange(r, 'b h n d -> b n (h d)')
#         return self.dropout(self.to_out(r))


class FastAttention(nn.Module):
    """ wuch15's Fastformer Attention module (Official) """

    def __init__(self, dim, dim_head, heads, dropout=0.1, initializer_range=0.02):
        super(FastAttention, self).__init__()

        self.initializer_range = initializer_range
        if dim % dim_head != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (dim, dim_head))
        self.attention_head_size = int(dim /dim_head)
        self.num_attention_heads = dim_head
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim = dim

        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.to_q_attn_logits = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.to_k_attn_logits = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weights)
        self.dropout = nn.Dropout(dropout)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask):
        """
        hidden_states -- [B, T, H]
        mask -- [B, T]
        """
        mask = mask.unsqueeze(1)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        mask = (1.0 - mask) * -10000.0

        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        # batch_size, num_head, seq_len
        query_for_score = self.to_q_attn_logits(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # add attention mask
        query_for_score += mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat

        query_key_score=(self.to_k_attn_logits(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)

        # add attention mask
        query_key_score +=mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer

        return self.dropout(weighted_value)


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

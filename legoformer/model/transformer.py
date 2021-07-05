"""
    Transformer class definition

    The implementation mainly follows the implementation found in the PyTorch
        with added support of pre-residual connection normalization.

    Resources used to develop this script:
        - https://github.com/jwang0306/transformer-pytorch
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.pff = nn.Sequential(
            nn.Linear(hidden_size, filter_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(filter_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, src):
        src = self.pff(src)

        return src


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, pre_lnorm, dropout):
        super(EncoderLayer, self).__init__()
        # self-attention part
        self.self_attn = nn.MultiheadAttention(hidden_size, n_head, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(hidden_size)

        # feed forward network part
        self.pff = PositionwiseFeedForward(hidden_size, filter_size, dropout)
        self.pff_norm = nn.LayerNorm(hidden_size)

        self.pre_lnorm = pre_lnorm

    def forward(self, src):
        if self.pre_lnorm:
            pre = self.self_attn_norm(src)
            src = src + self.self_attn(pre, pre, pre)[0]  # residual connection

            pre = self.pff_norm(src)
            src = src + self.pff(pre)  # residual connection
        else:
            src = self.self_attn_norm(src + self.self_attn(src, src, src)[0])  # residual connection + layerNorm
            src = self.pff_norm(src + self.pff(src))  # residual connection + layerNorm

        return src


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, dropout, n_layers, pre_lnorm):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_scale = hidden_size ** 0.5
        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_size, filter_size, n_head, pre_lnorm, dropout) for _ in range(n_layers)])
        self.pre_lnorm = pre_lnorm
        self.last_norm = nn.LayerNorm(hidden_size)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)

        if self.pre_lnorm:
            src = self.last_norm(src)

        return src


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, pre_lnorm, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, n_head, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(hidden_size)

        self.ed_self_attn = nn.MultiheadAttention(hidden_size, n_head, dropout=dropout)
        self.ed_self_attn_norm = nn.LayerNorm(hidden_size)

        # feed forward network part
        self.pff = PositionwiseFeedForward(hidden_size, filter_size, dropout)
        self.pff_norm = nn.LayerNorm(hidden_size)

        self.pre_lnorm = pre_lnorm

    def forward(self, enc_out, trg, trg_mask):
        if self.pre_lnorm:
            ris = self.self_attn_norm(trg)
            trg = trg + self.self_attn(ris, ris, ris, attn_mask=trg_mask)[0]

            ris = self.ed_self_attn_norm(trg)
            trg = trg + self.ed_self_attn(ris, enc_out, enc_out)[0]

            ris = self.pff_norm(trg)
            trg = trg + self.pff(ris)
        else:
            trg = self.self_attn_norm(trg + self.self_attn(trg, trg, trg, attn_mask=trg_mask)[0])
            trg = self.ed_self_attn_norm(trg + self.ed_self_attn(trg, enc_out, enc_out)[0])
            trg = self.pff_norm(trg + self.pff(trg))

        return trg


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, dropout, n_layers, pre_lnorm):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embed_scale = hidden_size ** 0.5
        self.layers = nn.ModuleList(
            [DecoderLayer(hidden_size, filter_size, n_head, pre_lnorm, dropout) for _ in range(n_layers)])
        self.pre_lnorm = pre_lnorm
        self.last_norm = nn.LayerNorm(hidden_size)

    def forward(self, enc_out, trg, trg_mask=None):
        for layer in self.layers:
            trg = layer(enc_out, trg, trg_mask)

        if self.pre_lnorm:
            trg = self.last_norm(trg)

        return trg


class Transformer(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, dropout, num_encoder_layers, num_decoder_layers,
                 pre_lnorm=True):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, dim_feedforward, nhead, dropout, num_encoder_layers, pre_lnorm)
        self.decoder = Decoder(d_model, dim_feedforward, nhead, dropout, num_decoder_layers, pre_lnorm)

    def forward(self, src, trg, trg_mask=None):
        enc_out = self.encoder(src)
        dec_out = self.decoder(enc_out, trg, trg_mask)

        return dec_out

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

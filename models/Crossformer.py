import torch
from einops import repeat
from math import ceil
from torch import nn

from layers.Crossformer.cross_decoder import Decoder
from layers.Crossformer.cross_embed import DSW_embedding
from layers.Crossformer.cross_encoder import Encoder


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.data_dim = configs.enc_in
        self.in_len = configs.seq_len
        self.out_len = configs.pred_len

        self.baseline = configs.baseline

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.in_len / configs.seg_len) * configs.seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / configs.seg_len) * configs.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(configs.seg_len, configs.d_model)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, (self.pad_in_len // configs.seg_len), configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(configs.e_layers, configs.merge_win, configs.d_model, configs.n_heads, configs.d_ff,
                               block_depth=1, dropout=configs.dropout, in_seg_num=(self.pad_in_len // configs.seg_len),
                               factor=configs.factor)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, (self.pad_out_len // configs.seg_len), configs.d_model))
        self.decoder = Decoder(configs.seg_len, configs.e_layers + 1, configs.d_model, configs.n_heads, configs.d_ff,
                               configs.dropout, out_seg_num=(self.pad_out_len // configs.seg_len),
                               factor=configs.factor)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.baseline:
            base = x_enc.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_enc.shape[0]
        if self.in_len_add != 0:
            x_enc = torch.cat((x_enc[:, :1, :].expand(-1, self.in_len_add, -1), x_enc), dim=1)

        x_enc = self.enc_value_embedding(x_enc)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)

        enc_out = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        return base + predict_y[:, :self.out_len, :]

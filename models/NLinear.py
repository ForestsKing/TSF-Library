import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs, individual=False):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.Linear.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        seq_last = x_enc[:, -1:, :].detach()
        x_enc = x_enc - seq_last

        if self.individual:
            output = torch.zeros([x_enc.size(0), self.pred_len, x_enc.size(2)], dtype=x_enc.dtype).to(x_enc.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x_enc[:, :, i])
        else:
            output = self.Linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        output = output + seq_last

        return output[:, -self.pred_len:, :]

import torch
import torch.nn.functional as F
from torch import nn

from layers.TimesNet.ConvBlock import Inception_Block


def FFT_for_Period(x, k=2):
    # xf 中是以 idx/length 为频率的复数, 复数的绝对值即为振幅, 假设振幅越大越重要
    # .rfft 即只返回共轭的右(正)半部分
    xf = torch.fft.rfft(x, dim=1)

    # 模型认为每个 bs 每个 c 的周期特征相同
    frequency_list = abs(xf).mean(0).mean(-1)

    # Freq[0]具有一些特殊含义，表示的是整个信号的能量值。
    # Freq[1] 才表示整个序列没有周期，频率为 1/length, 周期为 length
    frequency_list[0] = 0

    # 返回振幅最大的 k 个 idx
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    # 周期为 1/频率 = length/idx
    period = x.shape[1] // top_list

    # 模型认为每个 c 的周期特征相同，每个 bs 的周期特征不同
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = x[:, -(length - (self.seq_len + self.pred_len)):, :]
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)

            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)

        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
        return res

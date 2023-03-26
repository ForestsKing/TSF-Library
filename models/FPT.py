import torch
import torch.nn as nn
from einops import rearrange
from transformers import GPT2Model, GPT2Config


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.patch_size = configs.patch_size
        self.stride = configs.stride

        self.padding_patch_layer = nn.ReplicationPad1d((0, configs.stride))
        self.patch_num = (configs.seq_len - configs.patch_size) // configs.stride + 1
        self.patch_num += 1  # add configs.stride to length before

        if configs.pretrain:
            self.gpt2 = GPT2Model.from_pretrained('gpt2')
        else:
            self.gpt2 = GPT2Model(GPT2Config())
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, _, C = x_enc.shape

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x = rearrange(x_enc, 'B L C -> B C L')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'B C N P -> (B C) N P')

        outputs = self.in_layer(x)
        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = rearrange(outputs, '(B C) N P -> (B C) (N P)', B=B, C=C)
        outputs = self.out_layer(outputs)
        outputs = rearrange(outputs, '(B C) L -> B L C', B=B, C=C)

        outputs = outputs * stdev
        outputs = outputs + means
        return outputs

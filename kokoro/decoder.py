import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm

class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s).unsqueeze(-1)  # [B, 2F, 1]
        gamma, beta = torch.chunk(h, 2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class UpSample1d(nn.Module):
    def __init__(self, method='nearest'):
        super().__init__()
        self.method = method

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode=self.method)

class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, upsample=False, actv=nn.LeakyReLU(0.2), dropout=0.0):
        super().__init__()
        self.upsample = UpSample1d() if upsample else nn.Identity()
        self.learned_sc = dim_in != dim_out

        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, padding=1))
        self.norm2 = AdaIN1d(style_dim, dim_out)
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, padding=1))

        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, kernel_size=1, bias=False))
        self.actv = actv
        self.dropout = nn.Dropout(dropout)

    def _shortcut(self, x):
        x = self.upsample(x)
        return self.conv1x1(x) if self.learned_sc else x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(self.dropout(x))

        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        return (self._shortcut(x) + self._residual(x, s)) / math.sqrt(2)

class Decoder(nn.Module):
    def __init__(self, in_channels=64, style_dim=64, hidden_dims=[128, 64, 48], out_channels=1):
        super().__init__()
        blocks = []
        dim_in = in_channels
        for dim_out in hidden_dims:
            blocks.append(AdainResBlk1d(dim_in, dim_out, style_dim=style_dim, upsample=True))
            dim_in = dim_out

        self.res_blocks = nn.ModuleList(blocks)
        self.out_conv = weight_norm(nn.Conv1d(dim_in, out_channels, kernel_size=5, padding=2))

    def forward(self, x, s):
        for block in self.res_blocks:
            x = block(x, s)
        x = self.out_conv(x)
        return x


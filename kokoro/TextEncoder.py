import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)  # (B, T, C)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)  # (B, C, T)

class TextEncoder(nn.Module):
    def __init__(self, channels=64, kernel_size=5, depth=3, n_symbols=60, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.convs = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=1)),
                LayerNorm(channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_p),
            )
            for _ in range(depth)
        ])

        self.rnn = nn.GRU(channels, channels, num_layers=1, batch_first=True)

    def forward(self, x, input_lengths=None, mask=None):
        """
        x: LongTensor [B, T]
        input_lengths: LongTensor [B]
        mask: BoolTensor [B, T] where True = padding
        """
        x = self.embedding(x)  # [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(1), 0.0)

        for conv in self.convs:
            x = conv(x)
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(1), 0.0)

        x = x.transpose(1, 2)  # [B, T, C]

        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)

        x = x.transpose(1, 2)  # [B, C, T]
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(1), 0.0)

        return x

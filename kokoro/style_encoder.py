import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleStyleEncoder(nn.Module):
    def __init__(self, input_dim=80, style_dim=48, hidden_dim=128, num_layers=4):
        super(SimpleStyleEncoder, self).__init__()
        self.convs = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(input_dim if i == 0 else hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ) for i in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(hidden_dim, style_dim)

    def forward(self, x):
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.convs(x)
        x = self.pool(x).squeeze(-1)  # (B, hidden_dim)
        x = self.linear(x)  # (B, style_dim)
        return x

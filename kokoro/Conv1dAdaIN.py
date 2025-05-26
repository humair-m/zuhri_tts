class Conv1dAdaIN(nn.Module):
    def __init__(self, channels, style_dim, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        
        # Style modulation: maps style vector to scale and shift parameters
        self.style_scale = nn.Linear(style_dim, channels)
        self.style_shift = nn.Linear(style_dim, channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, style):
        """
        x: [B, C, T]
        style: [B, style_dim]
        """
        # Conv + InstanceNorm
        out = self.conv(x)
        out = self.norm(out)

        # Style modulation parameters
        scale = self.style_scale(style).unsqueeze(2)  # [B, C, 1]
        shift = self.style_shift(style).unsqueeze(2)  # [B, C, 1]

        # AdaIN: scale and shift normalized output by style params
        out = scale * out + shift
        out = self.relu(out)
        return out

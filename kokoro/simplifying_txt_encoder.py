class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))

    def forward(self, x, input_lengths=None, m=None):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        
        if m is not None:
            m = m.to(x.device).unsqueeze(1)
            x.masked_fill_(m, 0.0)

        for c in self.cnn:
            x = c(x)
            if m is not None:
                x.masked_fill_(m, 0.0)

        return x  # [B, channels, T]

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        for c in self.cnn:
            x = c(x)
        return x

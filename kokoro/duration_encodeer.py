class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)  # optional: embed input to model size
        self.style_proj = nn.Linear(sty_dim, d_model)

        # Use a single light BiLSTM (can replace with Conv1D or GRU)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, style, text_lengths=None, m=None):
        """
        x: [B, d_model, T]
        style: [B, sty_dim]
        """
        x = x.transpose(1, 2)  # [B, T, d_model]

        # Add style once
        style_proj = self.style_proj(style).unsqueeze(1)  # [B, 1, d_model]
        x = x + style_proj

        # Optional projection
        x = self.input_proj(x)

        # LSTM
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        # Layer norm and dropout
        x = self.norm(x)
        x = self.dropout(x)

        return x.transpose(1, 2)  # [B, d_model, T]

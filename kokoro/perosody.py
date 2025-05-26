class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, max_dur=50, dropout=0.1):
        super().__init__()
        # Use a single simple text encoder (conv or lightweight LSTM)
        self.text_encoder = DurationEncoder(style_dim, d_hid, dropout=dropout)
        
        # Single-layer BiLSTM for duration prediction (or replace with lightweight conv)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = nn.Linear(d_hid, max_dur)
        
        # Simplify F0 and Noise prediction blocks as lightweight conv + AdaIN or norm layers
        self.shared_lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        
        # Replace multiple AdainResBlk1d with simple Conv1d + AdaIN layers or fewer blocks
        self.F0_blocks = nn.Sequential(
            Conv1dAdaIN(d_hid, d_hid, style_dim, dropout),
            Conv1dAdaIN(d_hid, d_hid // 2, style_dim, dropout, upsample=True),
            Conv1dAdaIN(d_hid // 2, d_hid // 2, style_dim, dropout)
        )
        self.N_blocks = nn.Sequential(
            Conv1dAdaIN(d_hid, d_hid, style_dim, dropout),
            Conv1dAdaIN(d_hid, d_hid // 2, style_dim, dropout, upsample=True),
            Conv1dAdaIN(d_hid // 2, d_hid // 2, style_dim, dropout)
        )
        
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)
        
        # Predict duration with packed LSTM sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            d, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        x, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        duration = self.duration_proj(x)
        
        en = torch.matmul(d.transpose(1, 2), alignment)
        
        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        self.shared_lstm.flatten_parameters()
        x, _ = self.shared_lstm(x.transpose(1, 2))
        x = x.transpose(1, 2)

        F0 = self.F0_blocks(x, s)
        F0 = self.F0_proj(F0)

        N = self.N_blocks(x, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

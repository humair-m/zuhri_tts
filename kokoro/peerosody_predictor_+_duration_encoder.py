import torch
import torch.nn as nn
import torch.nn.functional as F

# Assumes AdainResBlk1d, AdaLayerNorm, LinearNorm, and JDCNet are already defined elsewhere

class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()

        self.text_encoder = DurationEncoder(
            sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout
        )

        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, max_dur)

        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1,
                              batch_first=True, bidirectional=True)

        self.F0 = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        ])

        self.N = nn.ModuleList([
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout)
        ])

        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1)

    def forward(self, texts, style, text_lengths, alignment, m):
        d = self.text_encoder(texts, style, text_lengths, m)

        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)
        m = m.to(text_lengths.device).unsqueeze(1)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Pad with zeros if needed
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]], device=x.device)
        x_pad[:, :x.shape[1], :] = x
        x = x_pad

        duration = self.duration_proj(F.dropout(x, 0.5, training=self.training))
        en = d.transpose(-1, -2) @ alignment

        return duration.squeeze(-1), en

    def F0Ntrain(self, x, s):
        x, _ = self.shared(x.transpose(-1, -2))  # shape: B x C x T → B x T x C

        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)

class DurationEncoder(nn.Module):
    def __init__(self, sty_dim, d_model, nlayers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(nn.LSTM(d_model + sty_dim, d_model // 2,
                                       num_layers=1, batch_first=True, bidirectional=True, dropout=dropout))
            self.layers.append(AdaLayerNorm(sty_dim, d_model))

        self.dropout = dropout

    def forward(self, x, style, text_lengths, m):
        masks = m.to(text_lengths.device)
        x = x.permute(2, 0, 1)  # T x B x C
        s = style.expand(x.shape[0], x.shape[1], -1)  # T x B x style_dim
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)  # B x T x (C + style)

        input_lengths = text_lengths.cpu().numpy()
        x = x.transpose(-1, -2)  # B x (C+style) x T

        for block in self.layers:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 0, 2)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, :x.shape[-1]] = x
                x = x_pad

        return x.transpose(-1, -2)

def load_F0_models(path):
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()
    return F0_model

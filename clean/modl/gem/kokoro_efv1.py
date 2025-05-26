import os
import os.path as osp
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
     
from munch import Munch
import yaml

class LinearNorm(torch.nn.Module):
    """Linear layer with Xavier uniform initialization."""
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class LayerNorm(nn.Module):
    """Layer Normalization for channel-first tensors (B, C, T)."""
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1) # [B, T, C]
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1) # [B, C, T]

class KokoroTextEncoder(nn.Module):
    """Text encoder with embeddings, CNNs, and a bidirectional LSTM."""
    def __init__(self, channels=256, kernel_size=5, depth=4, n_symbols=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ))

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(channels, channels)

    def forward(self, x, input_lengths=None, mask=None):
        x = self.embedding(x)  # [B, T_text, channels]
        x = x.transpose(1, 2)  # [B, channels, T_text]

        if mask is not None:
            mask_expanded = mask.to(x.device).unsqueeze(1)
            x.masked_fill_(mask_expanded, 0.0)
        
        for c in self.cnn:
            x = c(x)
            if mask is not None:
                x.masked_fill_(mask_expanded, 0.0)
                
        x = x.transpose(1, 2)  # [B, T_text, channels]

        if input_lengths is not None:
            input_lengths_cpu = input_lengths.cpu() # Must be on CPU for pack_padded_sequence
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths_cpu, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        
        if input_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        x = self.output_proj(x)
        return x

class VoicePacketEmbedding(nn.Module):
    """Combines voice and language embeddings for multi-speaker/lingual synthesis."""
    def __init__(self, num_voices=54, voice_dim=256):
        super().__init__()
        self.num_voices = num_voices
        self.voice_dim = voice_dim
        self.voice_embeddings = nn.Embedding(num_voices, voice_dim)
        self.language_embeddings = nn.Embedding(8, voice_dim // 4) # Assuming 8 languages
        self.proj = nn.Linear(voice_dim + voice_dim // 4, voice_dim)
        
    def forward(self, voice_id, language_id=None):
        voice_emb = self.voice_embeddings(voice_id) # [B, voice_dim] or [voice_dim]
        
        if language_id is not None:
            lang_emb = self.language_embeddings(language_id) # [B, voice_dim // 4] or [voice_dim // 4]

            # Ensure lang_emb has a batch dimension if input was scalar
            if lang_emb.dim() == 0: # If language_id was a scalar (e.g., int), embedding output is 0-dim
                lang_emb = lang_emb.unsqueeze(0) # Make it [1, D_lang]
            elif lang_emb.dim() == 1 and voice_emb.dim() == 2: # If lang_emb is [D_lang] and voice_emb is [B, D_voice]
                 lang_emb = lang_emb.unsqueeze(0) # Make it [1, D_lang] to match batch dimension for concat

            combined = torch.cat([voice_emb, lang_emb], dim=-1)
            return self.proj(combined)
            
        return voice_emb

class KokoroProsodyPredictor(nn.Module):
    """Predicts F0, energy, and duration from text features and voice embedding."""
    def __init__(self, text_dim=256, voice_dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.voice_proj = nn.Linear(voice_dim, hidden_dim)
        
        self.prosody_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.f0_head = nn.Linear(hidden_dim // 2, 1)
        self.energy_head = nn.Linear(hidden_dim // 2, 1)
        self.duration_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, text_features, voice_embedding):
        text_proj = self.text_proj(text_features)
        voice_proj = self.voice_proj(voice_embedding)
        
        # Expand voice embedding to match text sequence length
        if voice_proj.dim() == 0: # Handle scalar voice_embedding
            voice_proj = voice_proj.unsqueeze(0)
        if voice_proj.dim() == 1: # If voice_proj is [D], make it [1, D]
            voice_proj = voice_proj.unsqueeze(0)
        voice_proj = voice_proj.unsqueeze(1).expand(-1, text_proj.size(1), -1)
            
        combined = torch.cat([text_proj, voice_proj], dim=-1)
        prosody_features = self.prosody_net(combined)
        
        f0 = self.f0_head(prosody_features).squeeze(-1)
        energy = self.energy_head(prosody_features).squeeze(-1)
        duration = torch.sigmoid(self.duration_head(prosody_features)).squeeze(-1)
        
        return {
            'f0': f0,
            'energy': energy,
            'duration': duration,
            'prosody_features': prosody_features
        }

class KokoroDecoder(nn.Module):
    """Lightweight decoder for mel-spectrogram generation."""
    def __init__(self, 
                 text_dim=256, 
                 voice_dim=256, 
                 prosody_dim=128,
                 mel_dim=80, 
                 hidden_dim=512,
                 num_layers=6,
                 dropout=0.1):
        super().__init__()
        
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(text_dim + voice_dim + prosody_dim, hidden_dim)
        
        self.decoder_layers = nn.ModuleList([
            KokoroDecoderLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, mel_dim)
        )
        
        self.postnet = KokoroPostnet(mel_dim, hidden_dim // 4)
        
    def forward(self, text_features, voice_embedding, prosody_features, lengths=None):
        batch_size, seq_len, _ = text_features.shape
        
        # Expand voice embedding
        if voice_embedding.dim() == 0: # Handle scalar voice_embedding
            voice_embedding = voice_embedding.unsqueeze(0)
        if voice_embedding.dim() == 1: # If voice_embedding is [D], make it [1, D]
            voice_embedding = voice_embedding.unsqueeze(0)
        voice_embedding = voice_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            
        combined_features = torch.cat([text_features, voice_embedding, prosody_features], dim=-1)
        
        x = self.input_proj(combined_features)
        
        for layer in self.decoder_layers:
            x = layer(x, lengths)
            
        mel_before = self.output_proj(x)
        mel_after = self.postnet(mel_before) + mel_before
        
        return {
            'mel_before': mel_before,
            'mel_after': mel_after
        }

class KokoroDecoderLayer(nn.Module):
    """Single decoder layer with self-attention and feed-forward."""
    def __init__(self, hidden_dim, dropout=0.1, num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        attn_mask = None
        if lengths is not None:
            max_len = x.size(1)
            attn_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            
        residual = x
        x_attn, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(residual + self.dropout_layer(x_attn))
        
        residual = x
        x_ffn = self.ffn(x)
        x = self.norm2(residual + x_ffn)
        
        return x

class KokoroPostnet(nn.Module):
    """Postnet for mel-spectrogram refinement."""
    def __init__(self, mel_dim=80, hidden_dim=128, num_layers=5, kernel_size=5):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        self.convs = nn.ModuleList()
        
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(mel_dim, hidden_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.5)
            )
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5)
                )
            )
            
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(hidden_dim, mel_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(0.5)
            )
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        return x

class KokoroTTS(nn.Module):
    """Complete Kokoro-82M TTS model."""
    def __init__(self, 
                 n_symbols=256,
                 num_voices=54,
                 num_languages=8,
                 text_dim=256,
                 voice_dim=256,
                 prosody_dim=128,
                 mel_dim=80,
                 hidden_dim=512,
                 num_decoder_layers=6,
                 dropout=0.1):
        super().__init__()
        
        self.text_encoder = KokoroTextEncoder(
            channels=text_dim, 
            n_symbols=n_symbols,
            dropout=dropout
        )
        
        self.voice_embedding = VoicePacketEmbedding(
            num_voices=num_voices,
            voice_dim=voice_dim
        )
        
        self.prosody_predictor = KokoroProsodyPredictor(
            text_dim=text_dim,
            voice_dim=voice_dim,
            hidden_dim=prosody_dim * 2,
            dropout=dropout
        )
        
        self.decoder = KokoroDecoder(
            text_dim=text_dim,
            voice_dim=voice_dim,
            prosody_dim=prosody_dim,
            mel_dim=mel_dim,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
    def forward(self, text_tokens, voice_ids, language_ids=None, text_lengths=None):
        text_features = self.text_encoder(text_tokens, text_lengths)
        voice_embeddings = self.voice_embedding(voice_ids, language_ids)
        prosody_output = self.prosody_predictor(text_features, voice_embeddings)
        
        decoder_output = self.decoder(
            text_features, 
            voice_embeddings, 
            prosody_output['prosody_features'],
            text_lengths
        )
        
        return {
            'mel_before': decoder_output['mel_before'],
            'mel_after': decoder_output['mel_after'],
            'f0': prosody_output['f0'],
            'energy': prosody_output['energy'],
            'duration': prosody_output['duration']
        }
    
    def inference(self, text_tokens, voice_id, language_id=None):
        self.eval()
        with torch.no_grad():
            # Ensure voice_id is a 1-element tensor
            if isinstance(voice_id, int):
                voice_id_tensor = torch.tensor([voice_id], device=text_tokens.device)
            elif voice_id.dim() == 0:
                voice_id_tensor = voice_id.unsqueeze(0)
            else:
                voice_id_tensor = voice_id

            # Ensure language_id is a 1-element tensor, if provided
            language_id_tensor = None
            if language_id is not None:
                if isinstance(language_id, int):
                    language_id_tensor = torch.tensor([language_id], device=text_tokens.device)
                elif language_id.dim() == 0:
                    language_id_tensor = language_id.unsqueeze(0)
                else:
                    language_id_tensor = language_id
                
            # Ensure text_tokens has a batch dimension
            input_text_tokens = text_tokens.unsqueeze(0) if text_tokens.dim() == 1 else text_tokens
            
            return self.forward(
                input_text_tokens,
                voice_id_tensor,
                language_id_tensor,
                text_lengths=None
            )

def build_kokoro_model(config):
    """Build Kokoro-82M model with given configuration."""
    model = KokoroTTS(
        n_symbols=config.get('n_symbols', 256),
        num_voices=config.get('num_voices', 54),
        num_languages=config.get('num_languages', 8),
        text_dim=config.get('text_dim', 256),
        voice_dim=config.get('voice_dim', 256),
        prosody_dim=config.get('prosody_dim', 128),
        mel_dim=config.get('mel_dim', 80),
        hidden_dim=config.get('hidden_dim', 512),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        dropout=config.get('dropout', 0.1)
    )
    return model

class KokoroLoss(nn.Module):
    """Combined loss function for Kokoro-82M training."""
    def __init__(self, mel_weight=1.0, prosody_weight=0.1):
        super().__init__()
        self.mel_weight = mel_weight
        self.prosody_weight = prosody_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        mel_loss_before = self.l1_loss(predictions['mel_before'], targets['mel'])
        mel_loss_after = self.l1_loss(predictions['mel_after'], targets['mel'])
        mel_loss = mel_loss_before + mel_loss_after
        
        prosody_loss = 0.0
        if 'f0' in targets and predictions['f0'].shape == targets['f0'].shape:
            prosody_loss += self.mse_loss(predictions['f0'], targets['f0'])
        if 'energy' in targets and predictions['energy'].shape == targets['energy'].shape:
            prosody_loss += self.mse_loss(predictions['energy'], targets['energy'])
        if 'duration' in targets and predictions['duration'].shape == targets['duration'].shape:
            prosody_loss += self.mse_loss(predictions['duration'], targets['duration'])
            
        total_loss = self.mel_weight * mel_loss + self.prosody_weight * prosody_loss
        
        return {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'prosody_loss': prosody_loss
        }

def load_kokoro_checkpoint(model, checkpoint_path, device='cpu'):
    """Load Kokoro model checkpoint."""
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('step', 0)

def save_kokoro_checkpoint(model, optimizer, epoch, step, checkpoint_path):
    """Save Kokoro model checkpoint."""
    output_dir = osp.dirname(checkpoint_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }, checkpoint_path)

KOKORO_CONFIG = {
    'n_symbols': 256,
    'num_voices': 54,
    'num_languages': 8,
    'text_dim': 256,
    'voice_dim': 256,
    'prosody_dim': 128,
    'mel_dim': 80,
    'hidden_dim': 512,
    'num_decoder_layers': 6,
    'dropout': 0.1
}

"""
Kokoro-82M: Efficient Text-to-Speech System
A streamlined reimplementation of the Kokoro TTS architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from typing import Optional, Tuple, Dict, List
import numpy as np


class SpectralNormConv1d(nn.Module):
    """1D Convolution with spectral normalization"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.conv = spectral_norm(nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias))
    
    def forward(self, x):
        return self.conv(x)


class SpectralNormConv2d(nn.Module):
    """2D Convolution with spectral normalization"""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=bias))
    
    def forward(self, x):
        return self.conv(x)


class LayerNorm1d(nn.Module):
    """1D Layer Normalization"""
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # x: [B, C, T]
        x = x.transpose(1, -1)  # [B, T, C]
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)  # [B, C, T]


class AdaptiveInstanceNorm1d(nn.Module):
    """Adaptive Instance Normalization for style conditioning"""
    def __init__(self, num_features, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.style_proj = nn.Linear(style_dim, num_features * 2)
    
    def forward(self, x, style):
        # x: [B, C, T], style: [B, style_dim]
        h = self.style_proj(style)  # [B, C*2]
        h = h.view(h.size(0), h.size(1), 1)  # [B, C*2, 1]
        gamma, beta = torch.chunk(h, 2, dim=1)  # [B, C, 1] each
        return (1 + gamma) * self.norm(x) + beta


class ResidualBlock1d(nn.Module):
    """1D Residual Block with optional style conditioning"""
    def __init__(self, dim_in, dim_out, kernel_size=3, style_dim=None, 
                 activation=nn.LeakyReLU(0.2), dropout=0.1, upsample=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.upsample = upsample
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
        padding = (kernel_size - 1) // 2
        
        # Main path
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, kernel_size, padding=padding))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, kernel_size, padding=padding))
        
        # Style conditioning
        if style_dim:
            self.norm1 = AdaptiveInstanceNorm1d(dim_in, style_dim)
            self.norm2 = AdaptiveInstanceNorm1d(dim_out, style_dim)
        else:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_out, affine=True)
        
        # Skip connection
        if dim_in != dim_out:
            self.skip_conv = weight_norm(nn.Conv1d(dim_in, dim_out, 1, bias=False))
        else:
            self.skip_conv = nn.Identity()
        
        # Upsampling
        if upsample:
            self.upsample_conv = weight_norm(
                nn.ConvTranspose1d(dim_in, dim_in, 3, stride=2, padding=1, output_padding=1, groups=dim_in)
            )
            self.upsample_skip = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.upsample_conv = nn.Identity()
            self.upsample_skip = nn.Identity()
    
    def forward(self, x, style=None):
        skip = self.upsample_skip(x)
        skip = self.skip_conv(skip)
        
        # Main path
        if style is not None:
            out = self.norm1(x, style)
        else:
            out = self.norm1(x)
        
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.upsample_conv(out)
        
        if style is not None:
            out = self.norm2(out, style)
        else:
            out = self.norm2(out)
        
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        return (out + skip) / math.sqrt(2)  # Unit variance


class ResidualBlock2d(nn.Module):
    """2D Residual Block for mel-spectrogram processing"""
    def __init__(self, dim_in, dim_out, kernel_size=3, activation=nn.LeakyReLU(0.2), 
                 downsample=False, normalize=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.downsample = downsample
        self.activation = activation
        
        padding = (kernel_size - 1) // 2
        
        self.conv1 = SpectralNormConv2d(dim_in, dim_in, kernel_size, padding=padding)
        self.conv2 = SpectralNormConv2d(dim_in, dim_out, kernel_size, padding=padding)
        
        if normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Skip connection
        if dim_in != dim_out:
            self.skip_conv = SpectralNormConv2d(dim_in, dim_out, 1, bias=False)
        else:
            self.skip_conv = nn.Identity()
        
        # Downsampling
        if downsample:
            self.downsample_conv = SpectralNormConv2d(dim_in, dim_in, 3, stride=2, padding=1, groups=dim_in)
            self.downsample_skip = nn.AvgPool2d(2)
        else:
            self.downsample_conv = nn.Identity()
            self.downsample_skip = nn.Identity()
    
    def forward(self, x):
        skip = self.downsample_skip(x)
        skip = self.skip_conv(skip)
        
        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.downsample_conv(out)
        
        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        return (out + skip) / math.sqrt(2)


class TextEncoder(nn.Module):
    """Text encoder with CNN and LSTM layers"""
    def __init__(self, vocab_size, hidden_dim=512, kernel_size=5, n_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.cnn_layers.append(nn.Sequential(
                weight_norm(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=(kernel_size-1)//2)),
                LayerNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ))
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
    
    def forward(self, text_tokens, text_lengths=None):
        # text_tokens: [B, T]
        x = self.embedding(text_tokens)  # [B, T, hidden_dim]
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        
        # Apply CNN layers
        for cnn in self.cnn_layers:
            x = cnn(x)
        
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        
        # LSTM
        if text_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        
        if text_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        return x.transpose(1, 2)  # [B, hidden_dim, T]


class StyleEncoder(nn.Module):
    """Style encoder for extracting speaker embeddings from mel-spectrograms"""
    def __init__(self, mel_dim=80, style_dim=256, hidden_dim=512):
        super().__init__()
        
        layers = []
        current_dim = mel_dim
        
        # Initial conv
        layers.append(SpectralNormConv2d(1, hidden_dim // 8, 3, padding=1))
        current_dim = hidden_dim // 8
        
        # Downsampling blocks
        for i in range(4):
            next_dim = min(current_dim * 2, hidden_dim)
            layers.append(ResidualBlock2d(current_dim, next_dim, downsample=True))
            current_dim = next_dim
        
        # Global pooling and final projection
        layers.extend([
            nn.LeakyReLU(0.2),
            SpectralNormConv2d(current_dim, current_dim, 5, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.LeakyReLU(0.2)
        ])
        
        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Linear(current_dim, style_dim)
    
    def forward(self, mel_spec):
        # mel_spec: [B, mel_dim, T] -> [B, 1, mel_dim, T]
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)
        
        h = self.encoder(mel_spec)  # [B, hidden_dim, 1, 1]
        h = h.view(h.size(0), -1)  # [B, hidden_dim]
        style = self.projection(h)  # [B, style_dim]
        
        return style


class ProsodyPredictor(nn.Module):
    """Predicts prosody features (duration, pitch, energy)"""
    def __init__(self, hidden_dim=512, style_dim=256, max_duration=50):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Shared encoder
        self.encoder = nn.LSTM(hidden_dim + style_dim, hidden_dim // 2, 
                              batch_first=True, bidirectional=True)
        
        # Duration predictor
        self.duration_proj = nn.Linear(hidden_dim, max_duration)
        
        # F0 and energy predictors
        self.f0_blocks = nn.ModuleList([
            ResidualBlock1d(hidden_dim, hidden_dim, style_dim=style_dim),
            ResidualBlock1d(hidden_dim, hidden_dim // 2, style_dim=style_dim, upsample=True),
            ResidualBlock1d(hidden_dim // 2, hidden_dim // 2, style_dim=style_dim)
        ])
        
        self.energy_blocks = nn.ModuleList([
            ResidualBlock1d(hidden_dim, hidden_dim, style_dim=style_dim),
            ResidualBlock1d(hidden_dim, hidden_dim // 2, style_dim=style_dim, upsample=True),
            ResidualBlock1d(hidden_dim // 2, hidden_dim // 2, style_dim=style_dim)
        ])
        
        self.f0_proj = nn.Conv1d(hidden_dim // 2, 1, 1)
        self.energy_proj = nn.Conv1d(hidden_dim // 2, 1, 1)
    
    def forward(self, text_enc, style, text_lengths=None):
        # text_enc: [B, hidden_dim, T]
        batch_size, _, seq_len = text_enc.shape
        
        # Expand style to match sequence length
        style_expanded = style.unsqueeze(2).expand(-1, -1, seq_len)  # [B, style_dim, T]
        
        # Concatenate text encoding and style
        x = torch.cat([text_enc, style_expanded], dim=1)  # [B, hidden_dim + style_dim, T]
        x = x.transpose(1, 2)  # [B, T, hidden_dim + style_dim]
        
        # Shared encoder
        if text_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        self.encoder.flatten_parameters()
        x, _ = self.encoder(x)
        
        if text_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        # Duration prediction
        duration = self.duration_proj(F.dropout(x, 0.5, training=self.training))
        duration = F.softmax(duration, dim=-1)
        
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        
        # F0 prediction
        f0 = x
        for block in self.f0_blocks:
            f0 = block(f0, style)
        f0 = self.f0_proj(f0).squeeze(1)  # [B, T]
        
        # Energy prediction
        energy = x
        for block in self.energy_blocks:
            energy = block(energy, style)
        energy = self.energy_proj(energy).squeeze(1)  # [B, T]
        
        return duration, f0, energy


class MelDecoder(nn.Module):
    """Lightweight decoder for mel-spectrogram generation"""
    def __init__(self, hidden_dim=512, mel_dim=80, style_dim=256, n_blocks=8):
        super().__init__()
        
        # Input projection
        self.input_proj = weight_norm(nn.Conv1d(hidden_dim, hidden_dim, 1))
        
        # Decoder blocks with upsampling
        self.blocks = nn.ModuleList()
        current_dim = hidden_dim
        
        for i in range(n_blocks):
            upsample = i < 3  # Upsample first 3 blocks
            next_dim = max(current_dim // 2, mel_dim) if upsample else current_dim
            
            self.blocks.append(ResidualBlock1d(
                current_dim, next_dim, 
                style_dim=style_dim,
                upsample=upsample,
                dropout=0.1
            ))
            current_dim = next_dim
        
        # Output projection
        self.output_proj = weight_norm(nn.Conv1d(current_dim, mel_dim, 1))
    
    def forward(self, x, style):
        # x: [B, hidden_dim, T]
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x, style)
        
        mel = self.output_proj(x)  # [B, mel_dim, T]
        return mel


class VoicePacketEmbedding(nn.Module):
    """Voice packet system for multi-speaker synthesis"""
    def __init__(self, n_speakers=54, style_dim=256):
        super().__init__()
        self.n_speakers = n_speakers
        self.speaker_embeddings = nn.Embedding(n_speakers, style_dim)
        self.style_dim = style_dim
        
        # Initialize with normal distribution
        nn.init.normal_(self.speaker_embeddings.weight, 0, 0.1)
    
    def forward(self, speaker_ids=None, reference_mel=None, style_encoder=None):
        if speaker_ids is not None:
            # Use predefined speaker embedding
            return self.speaker_embeddings(speaker_ids)
        elif reference_mel is not None and style_encoder is not None:
            # Extract style from reference mel-spectrogram
            return style_encoder(reference_mel)
        else:
            raise ValueError("Either speaker_ids or reference_mel must be provided")


class Kokoro82M(nn.Module):
    """
    Kokoro-82M: Efficient Text-to-Speech System
    
    A streamlined TTS model with:
    - Text encoder with CNN + LSTM
    - Style encoder for speaker embeddings
    - Prosody predictor for duration, pitch, energy
    - Lightweight mel decoder
    - Voice packet system for multi-speaker synthesis
    """
    
    def __init__(self, 
                 vocab_size=1000,
                 hidden_dim=512,
                 style_dim=256,
                 mel_dim=80,
                 n_speakers=54,
                 max_duration=50):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.mel_dim = mel_dim
        
        # Core components
        self.text_encoder = TextEncoder(vocab_size, hidden_dim)
        self.style_encoder = StyleEncoder(mel_dim, style_dim, hidden_dim)
        self.prosody_predictor = ProsodyPredictor(hidden_dim, style_dim, max_duration)
        self.mel_decoder = MelDecoder(hidden_dim, mel_dim, style_dim)
        self.voice_packets = VoicePacketEmbedding(n_speakers, style_dim)
        
        # Additional components for training
        self.duration_loss = nn.MSELoss()
        self.mel_loss = nn.L1Loss()
        self.f0_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()
    
    def forward(self, 
                text_tokens: torch.Tensor,
                text_lengths: Optional[torch.Tensor] = None,
                speaker_ids: Optional[torch.Tensor] = None,
                reference_mel: Optional[torch.Tensor] = None,
                target_duration: Optional[torch.Tensor] = None,
                target_f0: Optional[torch.Tensor] = None,
                target_energy: Optional[torch.Tensor] = None,
                target_mel: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size = text_tokens.size(0)
        
        # Encode text
        text_enc = self.text_encoder(text_tokens, text_lengths)  # [B, hidden_dim, T]
        
        # Get style embedding
        style = self.voice_packets(speaker_ids, reference_mel, self.style_encoder)  # [B, style_dim]
        
        # Predict prosody
        duration_pred, f0_pred, energy_pred = self.prosody_predictor(text_enc, style, text_lengths)
        
        # Generate mel-spectrogram
        mel_pred = self.mel_decoder(text_enc, style)
        
        outputs = {
            'mel_pred': mel_pred,
            'duration_pred': duration_pred,
            'f0_pred': f0_pred,
            'energy_pred': energy_pred,
            'style': style
        }
        
        # Calculate losses during training
        if self.training and target_mel is not None:
            losses = {}
            
            # Mel reconstruction loss
            losses['mel_loss'] = self.mel_loss(mel_pred, target_mel)
            
            # Prosody losses
            if target_duration is not None:
                losses['duration_loss'] = self.duration_loss(duration_pred, target_duration)
            if target_f0 is not None:
                losses['f0_loss'] = self.f0_loss(f0_pred, target_f0)
            if target_energy is not None:
                losses['energy_loss'] = self.energy_loss(energy_pred, target_energy)
            
            # Total loss
            total_loss = losses['mel_loss']
            if 'duration_loss' in losses:
                total_loss += losses['duration_loss']
            if 'f0_loss' in losses:
                total_loss += 0.1 * losses['f0_loss']
            if 'energy_loss' in losses:
                total_loss += 0.1 * losses['energy_loss']
            
            losses['total_loss'] = total_loss
            outputs['losses'] = losses
        
        return outputs
    
    def inference(self, 
                  text_tokens: torch.Tensor,
                  speaker_id: int = 0,
                  reference_mel: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Inference mode for generating speech
        """
        self.eval()
        
        with torch.no_grad():
            if reference_mel is not None:
                speaker_ids = None
            else:
                speaker_ids = torch.tensor([speaker_id], device=text_tokens.device)
            
            outputs = self.forward(
                text_tokens=text_tokens.unsqueeze(0) if text_tokens.dim() == 1 else text_tokens,
                speaker_ids=speaker_ids,
                reference_mel=reference_mel
            )
            
            return outputs['mel_pred']
    
    def get_model_size(self):
        """Calculate model size in millions of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params / 1e6


# Utility functions
def create_model(config: dict = None) -> Kokoro82M:
    """Create a Kokoro-82M model with default or custom configuration"""
    default_config = {
        'vocab_size': 1000,
        'hidden_dim': 512,
        'style_dim': 256,
        'mel_dim': 80,
        'n_speakers': 54,
        'max_duration': 50
    }
    
    if config:
        default_config.update(config)
    
    model = Kokoro82M(**default_config)
    
    print(f"Created Kokoro-82M model with {model.get_model_size():.1f}M parameters")
    return model


def load_checkpoint(model: Kokoro82M, checkpoint_path: str, device: str = 'cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'epoch' in checkpoint:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model()
    
    # Example forward pass
    batch_size = 2
    seq_len = 100
    
    text_tokens = torch.randint(0, 1000, (batch_size, seq_len))
    text_lengths = torch.tensor([seq_len, seq_len-10])
    speaker_ids = torch.tensor([0, 1])
    
    # Training mode
    model.train()
    target_mel = torch.randn(batch_size, 80, seq_len)
    target_duration = torch.randn(batch_size, seq_len, 50)
    
    outputs = model(
        text_tokens=text_tokens,
        text_lengths=text_lengths,
        speaker_ids=speaker_ids,
        target_mel=target_mel,
        target_duration=target_duration
    )
    
    print("Training outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v.item():.4f}")
    
    # Inference mode
    mel_output = model.inference(text_tokens[0], speaker_id=0)
    print(f"\nInference output shape: {mel_output.shape}")

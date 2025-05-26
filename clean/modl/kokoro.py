# -*- coding: utf-8 -*-
"""
Kokoro-82M Style Decoder-Only TTS Model
Simplified non-diffusion model focused on efficient text-to-speech generation.
"""

import os
import math
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import yaml
from munch import Munch


# ================== Basic Building Blocks ==================

class LayerNorm(nn.Module):
    """1D Layer Normalization for sequence data."""
    
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class LinearNorm(nn.Module):
    """Linear layer with Xavier initialization."""
    
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = 'linear'):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    """1D Convolution with normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, 
                 stride: int = 1, padding: Optional[int] = None, dilation: int = 1, 
                 bias: bool = True, w_init_gain: str = 'linear'):
        super().__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.conv(signal)


# ================== Text Encoder ==================

class TextEncoder(nn.Module):
    """Simplified text encoder with embedding, CNN, and projection layers."""
    
    def __init__(self, n_vocab: int, channels: int, kernel_size: int, depth: int, 
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, channels)
        
        # CNN layers
        self.convolutions = nn.ModuleList()
        for _ in range(depth):
            conv_layer = nn.Sequential(
                ConvNorm(channels, channels, kernel_size=kernel_size, 
                        stride=1, padding=(kernel_size - 1) // 2,
                        dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convolutions.append(conv_layer)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, 
                           bidirectional=True, dropout=dropout)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        # Embedding
        x = self.embedding(x).transpose(1, 2)  # [B, C, T]
        
        # CNN layers
        for conv in self.convolutions:
            x = conv(x)
        
        x = x.transpose(1, 2)  # [B, T, C]
        
        # LSTM
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs.transpose(1, 2)  # [B, C, T]

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)
        
        for conv in self.convolutions:
            x = conv(x)
            
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs.transpose(1, 2)


# ================== Duration Predictor ==================

class DurationPredictor(nn.Module):
    """Lightweight duration predictor."""
    
    def __init__(self, in_channels: int, hidden_channels: int = 256, 
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = ConvNorm(in_channels, hidden_channels, kernel_size)
        self.norm1 = LayerNorm(hidden_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = ConvNorm(hidden_channels, hidden_channels, kernel_size)
        self.norm2 = LayerNorm(hidden_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x * x_mask)
        x = torch.relu(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x * x_mask)
        x = torch.relu(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        
        x = self.proj(x * x_mask)
        return x * x_mask


# ================== Mel Decoder ==================

class ResidualBlock1D(nn.Module):
    """1D Residual block for the decoder."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, 
                                          padding=dilation, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, 
                                          padding=dilation, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual


class MelDecoder(nn.Module):
    """Mel-spectrogram decoder - the main generator."""
    
    def __init__(self, in_channels: int = 512, out_channels: int = 80,
                 hidden_channels: int = 512, kernel_size: int = 3,
                 n_blocks: int = 6, upsample_rates: List[int] = [8, 8, 2, 2]):
        super().__init__()
        
        self.n_blocks = n_blocks
        self.upsample_rates = upsample_rates
        
        # Pre-conv
        self.pre_conv = weight_norm(nn.Conv1d(in_channels, hidden_channels, 7, padding=3))
        
        # Upsampling layers
        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        
        for i, upsample_rate in enumerate(upsample_rates):
            # Upsample layer
            self.upsamples.append(weight_norm(
                nn.ConvTranspose1d(
                    hidden_channels // (2**i),
                    hidden_channels // (2**(i+1)),
                    upsample_rate * 2,
                    stride=upsample_rate,
                    padding=upsample_rate // 2 + upsample_rate % 2,
                    output_padding=upsample_rate % 2
                )
            ))
            
            # Residual blocks
            ch = hidden_channels // (2**(i+1))
            for j in range(n_blocks):
                self.resblocks.append(ResidualBlock1D(ch, kernel_size, dilation=3**j))
        
        # Post-conv
        self.post_conv = weight_norm(nn.Conv1d(ch, out_channels, 7, padding=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_conv(x)
        
        for i, upsample in enumerate(self.upsamples):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            
            # Apply residual blocks
            xs = None
            for j in range(self.n_blocks):
                if xs is None:
                    xs = self.resblocks[i * self.n_blocks + j](x)
                else:
                    xs += self.resblocks[i * self.n_blocks + j](x)
            x = xs / self.n_blocks
        
        x = F.leaky_relu(x)
        x = self.post_conv(x)
        return x


# ================== Length Regulator ==================

class LengthRegulator(nn.Module):
    """Regulates length using duration predictions."""
    
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, duration: torch.Tensor, 
                max_len: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] - encoded text features
            duration: [B, T] - duration for each text token
            max_len: maximum output length
        """
        batch_size, channels, text_len = x.shape
        
        if max_len is None:
            max_len = duration.sum(dim=1).max().item()
        
        # Create output tensor
        output = torch.zeros(batch_size, channels, max_len, 
                           dtype=x.dtype, device=x.device)
        
        for batch_idx in range(batch_size):
            current_pos = 0
            for text_idx in range(text_len):
                dur = int(duration[batch_idx, text_idx].item())
                if dur > 0 and current_pos + dur <= max_len:
                    output[batch_idx, :, current_pos:current_pos + dur] = \
                        x[batch_idx, :, text_idx].unsqueeze(-1)
                    current_pos += dur
                    
        return output


# ================== Main Kokoro Model ==================

class KokoroTTS(nn.Module):
    """Kokoro-82M style decoder-only TTS model."""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Config parameters
        self.n_vocab = config.get('n_vocab', 1000)
        self.text_channels = config.get('text_channels', 512)
        self.hidden_channels = config.get('hidden_channels', 512)
        self.n_mel_channels = config.get('n_mel_channels', 80)
        
        # Components
        self.text_encoder = TextEncoder(
            n_vocab=self.n_vocab,
            channels=self.text_channels,
            kernel_size=config.get('text_kernel_size', 5),
            depth=config.get('text_depth', 4),
            dropout=config.get('dropout', 0.1)
        )
        
        self.duration_predictor = DurationPredictor(
            in_channels=self.text_channels,
            hidden_channels=config.get('duration_hidden', 256),
            kernel_size=config.get('duration_kernel_size', 3),
            dropout=config.get('dropout', 0.1)
        )
        
        self.length_regulator = LengthRegulator()
        
        # Projection layer to decoder input size
        self.text_to_decoder = nn.Conv1d(self.text_channels, self.hidden_channels, 1)
        
        self.mel_decoder = MelDecoder(
            in_channels=self.hidden_channels,
            out_channels=self.n_mel_channels,
            hidden_channels=self.hidden_channels,
            n_blocks=config.get('decoder_blocks', 6),
            upsample_rates=config.get('upsample_rates', [8, 8, 2, 2])
        )

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor,
                mel_targets: Optional[torch.Tensor] = None,
                duration_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        
        # Create text mask
        text_mask = self.sequence_mask(text_lengths, text.size(1)).unsqueeze(1).float()
        
        # Encode text
        text_encoded = self.text_encoder(text, text_lengths)  # [B, C, T]
        
        # Predict durations
        log_duration_pred = self.duration_predictor(text_encoded, text_mask)
        duration_pred = torch.exp(log_duration_pred) - 1.0
        
        # Use ground truth durations during training if available
        if duration_targets is not None:
            duration = duration_targets
        else:
            duration = duration_pred.squeeze(1)
        
        # Length regulation
        text_upsampled = self.length_regulator(text_encoded, duration)
        
        # Project to decoder input size
        decoder_input = self.text_to_decoder(text_upsampled)
        
        # Generate mel-spectrogram
        mel_pred = self.mel_decoder(decoder_input)
        
        return {
            'mel_pred': mel_pred,
            'duration_pred': duration_pred.squeeze(1),
            'text_encoded': text_encoded,
            'text_mask': text_mask
        }

    def inference(self, text: torch.Tensor, duration_scale: float = 1.0) -> torch.Tensor:
        """Inference mode."""
        self.eval()
        
        with torch.no_grad():
            # Encode text
            text_encoded = self.text_encoder.inference(text)  # [B, C, T]
            
            # Create dummy mask for single sequence
            text_mask = torch.ones(1, 1, text.size(1), device=text.device)
            
            # Predict durations
            log_duration_pred = self.duration_predictor(text_encoded, text_mask)
            duration_pred = torch.exp(log_duration_pred) - 1.0
            duration_pred = duration_pred.squeeze(1) * duration_scale
            
            # Ensure minimum duration
            duration_pred = torch.clamp(duration_pred, min=1.0)
            
            # Length regulation
            text_upsampled = self.length_regulator(text_encoded, duration_pred)
            
            # Project to decoder input size
            decoder_input = self.text_to_decoder(text_upsampled)
            
            # Generate mel-spectrogram
            mel_pred = self.mel_decoder(decoder_input)
            
            return mel_pred.squeeze(0)

    @staticmethod
    def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
        """Create sequence mask."""
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


# ================== Loss Functions ==================

class KokoroLoss:
    """Loss functions for Kokoro TTS training."""
    
    def __init__(self, mel_loss_weight: float = 1.0, duration_loss_weight: float = 0.1):
        self.mel_loss_weight = mel_loss_weight
        self.duration_loss_weight = duration_loss_weight

    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        
        # Mel reconstruction loss
        mel_loss = F.mse_loss(predictions['mel_pred'], targets['mel_target'])
        
        # Duration loss (if targets available)  
        duration_loss = torch.tensor(0.0, device=predictions['mel_pred'].device)
        if 'duration_target' in targets:
            duration_loss = F.mse_loss(
                predictions['duration_pred'], 
                targets['duration_target']
            )
        
        # Total loss
        total_loss = (self.mel_loss_weight * mel_loss + 
                     self.duration_loss_weight * duration_loss)
        
        return {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'duration_loss': duration_loss
        }


# ================== Model Factory ==================

def create_kokoro_model(config_path: Optional[str] = None) -> KokoroTTS:
    """Create Kokoro TTS model."""
    
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config for ~82M parameters
        config = {
            'n_vocab': 1000,
            'text_channels': 512,
            'hidden_channels': 512,
            'n_mel_channels': 80,
            'text_kernel_size': 5,
            'text_depth': 4,
            'duration_hidden': 256,
            'duration_kernel_size': 3,
            'decoder_blocks': 6,
            'upsample_rates': [8, 8, 2, 2],
            'dropout': 0.1
        }
    
    model = KokoroTTS(config)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ================== Example Usage ==================

def example_usage():
    """Example of how to use the Kokoro model."""
    
    # Create model
    model = create_kokoro_model()
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Example training forward pass
    batch_size = 4
    max_text_len = 100
    max_mel_len = 800
    
    # Example inputs
    text = torch.randint(0, 1000, (batch_size, max_text_len))
    text_lengths = torch.randint(50, max_text_len, (batch_size,))
    mel_targets = torch.randn(batch_size, 80, max_mel_len)
    duration_targets = torch.randint(1, 10, (batch_size, max_text_len)).float()
    
    # Training forward pass
    model.train()
    outputs = model(text, text_lengths, mel_targets, duration_targets)
    
    print("Training outputs:")
    print(f"  Mel pred shape: {outputs['mel_pred'].shape}")
    print(f"  Duration pred shape: {outputs['duration_pred'].shape}")
    
    # Inference
    model.eval()
    single_text = torch.randint(0, 1000, (1, 50))
    mel_output = model.inference(single_text, duration_scale=1.0)
    
    print(f"\nInference output shape: {mel_output.shape}")
    
    # Compute loss
    loss_fn = KokoroLoss()
    targets = {
        'mel_target': mel_targets,
        'duration_target': duration_targets
    }
    losses = loss_fn.compute_loss(outputs, targets)
    
    print(f"\nLosses:")
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print(f"  Mel loss: {losses['mel_loss'].item():.4f}")
    print(f"  Duration loss: {losses['duration_loss'].item():.4f}")


if __name__ == "__main__":
    example_usage()

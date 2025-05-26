import math
import torch
from torch import nn
from typing import Optional, Any, Callable, Union
from torch import Tensor
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as audio_F
import warnings

import random
random.seed(0)


def _get_activation_fn(activ: str) -> Callable[[Tensor], Tensor]:
    """
    Enhanced activation function factory with modern activations.
    
    Args:
        activ: Activation function name
        
    Returns:
        Activation function
    """
    if activ == 'relu':
        return nn.ReLU(inplace=True)
    elif activ == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activ == 'swish' or activ == 'silu':
        return nn.SiLU(inplace=True)  # Modern implementation
    elif activ == 'gelu':
        return nn.GELU()
    elif activ == 'mish':
        return nn.Mish(inplace=True)
    elif activ == 'elu':
        return nn.ELU(inplace=True)
    elif activ == 'prelu':
        return nn.PReLU()
    else:
        available_activs = ['relu', 'lrelu', 'swish', 'silu', 'gelu', 'mish', 'elu', 'prelu']
        raise RuntimeError(f'Unexpected activation type {activ}, expected one of {available_activs}')


class LinearNorm(nn.Module):
    """
    Enhanced linear layer with modern initialization and optional features.
    """
    
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 bias: bool = True, 
                 w_init_gain: str = 'linear',
                 dropout: float = 0.0,
                 activation: Optional[str] = None,
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 spectral_norm: bool = False):
        """
        Initialize LinearNorm layer.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            bias: Whether to use bias
            w_init_gain: Weight initialization gain
            dropout: Dropout rate
            activation: Optional activation function
            batch_norm: Whether to apply batch normalization
            layer_norm: Whether to apply layer normalization
            spectral_norm: Whether to apply spectral normalization
        """
        super(LinearNorm, self).__init__()
        
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        
        # Enhanced weight initialization
        if w_init_gain == 'linear':
            nn.init.xavier_uniform_(self.linear_layer.weight)
        elif w_init_gain == 'relu':
            nn.init.kaiming_uniform_(self.linear_layer.weight, nonlinearity='relu')
        elif w_init_gain == 'leaky_relu':
            nn.init.kaiming_uniform_(self.linear_layer.weight, nonlinearity='leaky_relu')
        else:
            try:
                gain = nn.init.calculate_gain(w_init_gain)
                nn.init.xavier_uniform_(self.linear_layer.weight, gain=gain)
            except ValueError:
                nn.init.xavier_uniform_(self.linear_layer.weight)
        
        # Initialize bias
        if bias:
            nn.init.zeros_(self.linear_layer.bias)
        
        # Apply spectral normalization if requested
        if spectral_norm:
            self.linear_layer = nn.utils.spectral_norm(self.linear_layer)
        
        # Optional normalization layers
        self.batch_norm = nn.BatchNorm1d(out_dim) if batch_norm else None
        self.layer_norm = nn.LayerNorm(out_dim) if layer_norm else None
        
        # Optional activation and dropout
        self.activation = _get_activation_fn(activation) if activation else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional normalization, activation, and dropout."""
        x = self.linear_layer(x)
        
        # Apply normalization
        if self.batch_norm is not None:
            # Handle different input shapes for batch norm
            if x.dim() == 3:  # [B, T, C]
                x = x.transpose(1, 2)
                x = self.batch_norm(x)
                x = x.transpose(1, 2)
            else:  # [B, C]
                x = self.batch_norm(x)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        # Apply activation
        if self.activation is not None:
            x = self.activation(x)
        
        # Apply dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class ConvNorm(nn.Module):
    """
    Enhanced 1D convolution layer with modern features.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 1, 
                 stride: int = 1,
                 padding: Optional[int] = None, 
                 dilation: int = 1, 
                 bias: bool = True, 
                 w_init_gain: str = 'linear', 
                 param: Optional[Any] = None,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 activation: Optional[str] = None,
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 spectral_norm: bool = False):
        """
        Initialize ConvNorm layer.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Padding size (auto-calculated if None)
            dilation: Dilation rate
            bias: Whether to use bias
            w_init_gain: Weight initialization gain
            param: Additional parameter for gain calculation
            groups: Number of groups for grouped convolution
            padding_mode: Padding mode ('zeros', 'reflect', 'replicate', 'circular')
            activation: Optional activation function
            dropout: Dropout rate
            batch_norm: Whether to apply batch normalization
            spectral_norm: Whether to apply spectral normalization
        """
        super(ConvNorm, self).__init__()
        
        # Auto-calculate padding for 'same' convolution
        if padding is None:
            if kernel_size % 2 == 1:
                padding = int(dilation * (kernel_size - 1) / 2)
            else:
                padding = 0
                warnings.warn(f"Even kernel_size {kernel_size} with auto-padding may cause size mismatch")

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            bias=bias, groups=groups, padding_mode=padding_mode
        )

        # Enhanced weight initialization
        if w_init_gain == 'linear':
            nn.init.xavier_uniform_(self.conv.weight)
        elif w_init_gain == 'relu':
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        elif w_init_gain == 'leaky_relu':
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='leaky_relu', a=0.2)
        else:
            try:
                gain = nn.init.calculate_gain(w_init_gain, param=param)
                nn.init.xavier_uniform_(self.conv.weight, gain=gain)
            except (ValueError, TypeError):
                nn.init.xavier_uniform_(self.conv.weight)
        
        # Initialize bias
        if bias:
            nn.init.zeros_(self.conv.bias)
        
        # Apply spectral normalization if requested
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        
        # Optional layers
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        self.activation = _get_activation_fn(activation) if activation else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, signal: Tensor) -> Tensor:
        """Forward pass with optional normalization, activation, and dropout."""
        conv_signal = self.conv(signal)
        
        if self.batch_norm is not None:
            conv_signal = self.batch_norm(conv_signal)
        
        if self.activation is not None:
            conv_signal = self.activation(conv_signal)
        
        if self.dropout is not None:
            conv_signal = self.dropout(conv_signal)
        
        return conv_signal


class CausalConv(nn.Module):  # Fixed typo: Causual -> Causal
    """
    Enhanced causal convolution for autoregressive models.
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 1, 
                 stride: int = 1, 
                 padding: int = 1, 
                 dilation: int = 1, 
                 bias: bool = True, 
                 w_init_gain: str = 'linear', 
                 param: Optional[Any] = None,
                 groups: int = 1,
                 activation: Optional[str] = None,
                 dropout: float = 0.0,
                 batch_norm: bool = False):
        """
        Initialize CausalConv layer.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Base padding (will be doubled for causal)
            dilation: Dilation rate
            bias: Whether to use bias
            w_init_gain: Weight initialization gain
            param: Additional parameter for gain calculation
            groups: Number of groups for grouped convolution
            activation: Optional activation function
            dropout: Dropout rate
            batch_norm: Whether to apply batch normalization
        """
        super(CausalConv, self).__init__()
        
        # Calculate causal padding
        if padding is None:
            self.padding = int(dilation * (kernel_size - 1))
        else:
            self.padding = padding * 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=self.padding,
            dilation=dilation,
            bias=bias, groups=groups
        )

        # Enhanced weight initialization
        self._initialize_weights(w_init_gain, param)
        
        # Optional layers
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        self.activation = _get_activation_fn(activation) if activation else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _initialize_weights(self, w_init_gain: str, param: Optional[Any]):
        """Initialize convolution weights."""
        if w_init_gain == 'linear':
            nn.init.xavier_uniform_(self.conv.weight)
        elif w_init_gain == 'relu':
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        elif w_init_gain == 'leaky_relu':
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='leaky_relu', a=0.2)
        else:
            try:
                gain = nn.init.calculate_gain(w_init_gain, param=param)
                nn.init.xavier_uniform_(self.conv.weight, gain=gain)
            except (ValueError, TypeError):
                nn.init.xavier_uniform_(self.conv.weight)
        
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with causal masking."""
        x = self.conv(x)
        
        # Remove future information for causality
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        
        if self.activation is not None:
            x = self.activation(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class CausalBlock(nn.Module):  # Fixed typo: CausualBlock -> CausalBlock
    """
    Enhanced causal convolutional block with residual connections.
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 n_conv: int = 3, 
                 dropout_p: float = 0.2, 
                 activ: str = 'lrelu',
                 use_layer_norm: bool = False,
                 use_channel_shuffle: bool = False):
        """
        Initialize CausalBlock.
        
        Args:
            hidden_dim: Hidden dimension
            n_conv: Number of convolution layers
            dropout_p: Dropout probability
            activ: Activation function
            use_layer_norm: Whether to use layer normalization instead of batch norm
            use_channel_shuffle: Whether to use channel shuffle for efficiency
        """
        super(CausalBlock, self).__init__()
        
        self.use_channel_shuffle = use_channel_shuffle
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, 
                          dropout_p=dropout_p, use_layer_norm=use_layer_norm)
            for i in range(n_conv)
        ])
        
        # Optional channel shuffle
        if use_channel_shuffle:
            self.channel_shuffle = nn.ChannelShuffle(groups=min(8, hidden_dim // 8))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connections."""
        for i, block in enumerate(self.blocks):
            residual = x
            x = block(x)
            
            # Add residual connection
            if x.shape == residual.shape:
                x = x + residual
            
            # Apply channel shuffle periodically
            if self.use_channel_shuffle and i % 2 == 1:
                x = self.channel_shuffle(x)
        
        return x

    def _get_conv(self, 
                  hidden_dim: int, 
                  dilation: int, 
                  activ: str = 'lrelu', 
                  dropout_p: float = 0.2,
                  use_layer_norm: bool = False) -> nn.Sequential:
        """Create a single convolution block."""
        layers = [
            CausalConv(hidden_dim, hidden_dim, kernel_size=3, 
                      padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
        ]
        
        # Choose normalization
        if use_layer_norm:
            layers.append(nn.GroupNorm(1, hidden_dim))  # LayerNorm equivalent for conv
        else:
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        layers.extend([
            nn.Dropout(p=dropout_p),
            CausalConv(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ])
        
        return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """
    Enhanced convolutional block with modern features.
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 n_conv: int = 3, 
                 dropout_p: float = 0.2, 
                 activ: str = 'relu',
                 use_se_block: bool = False,
                 use_cbam: bool = False,
                 expansion_ratio: float = 1.0):
        """
        Initialize ConvBlock.
        
        Args:
            hidden_dim: Hidden dimension
            n_conv: Number of convolution layers
            dropout_p: Dropout probability
            activ: Activation function
            use_se_block: Whether to use Squeeze-and-Excitation
            use_cbam: Whether to use CBAM attention
            expansion_ratio: Channel expansion ratio for bottleneck
        """
        super().__init__()
        
        self._n_groups = max(1, min(8, hidden_dim // 8))  # Adaptive group size
        self.use_se_block = use_se_block
        self.use_cbam = use_cbam
        
        self.blocks = nn.ModuleList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, 
                          dropout_p=dropout_p, expansion_ratio=expansion_ratio)
            for i in range(n_conv)
        ])
        
        # Attention mechanisms
        if use_se_block:
            self.se_block = SEBlock(hidden_dim)
        
        if use_cbam:
            self.cbam = CBAM(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with attention and residual connections."""
        for i, block in enumerate(self.blocks):
            residual = x
            x = block(x)
            
            # Add residual connection
            if x.shape == residual.shape:
                x = x + residual
        
        # Apply attention mechanisms
        if self.use_se_block:
            x = self.se_block(x)
        
        if self.use_cbam:
            x = self.cbam(x)
        
        return x

    def _get_conv(self, 
                  hidden_dim: int, 
                  dilation: int, 
                  activ: str = 'relu', 
                  dropout_p: float = 0.2,
                  expansion_ratio: float = 1.0) -> nn.Sequential:
        """Create a single convolution block with optional bottleneck."""
        expanded_dim = int(hidden_dim * expansion_ratio)
        
        layers = []
        
        # Expansion layer if needed
        if expansion_ratio != 1.0:
            layers.extend([
                ConvNorm(hidden_dim, expanded_dim, kernel_size=1),
                _get_activation_fn(activ),
            ])
            input_dim = expanded_dim
        else:
            input_dim = hidden_dim
        
        # Main convolution layers
        layers.extend([
            ConvNorm(input_dim, input_dim, kernel_size=3, 
                    padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.GroupNorm(num_groups=self._n_groups, num_channels=input_dim),
            nn.Dropout(p=dropout_p),
            ConvNorm(input_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ])
        
        return nn.Sequential(*layers)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class ChannelAttention(nn.Module):
    """Channel attention component of CBAM."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1)


class SpatialAttention(nn.Module):
    """Spatial attention component of CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class LocationLayer(nn.Module):
    """
    Enhanced location layer for attention mechanisms.
    """
    
    def __init__(self, 
                 attention_n_filters: int, 
                 attention_kernel_size: int,
                 attention_dim: int,
                 dropout: float = 0.0):
        """
        Initialize LocationLayer.
        
        Args:
            attention_n_filters: Number of attention filters
            attention_kernel_size: Attention kernel size
            attention_dim: Attention dimension
            dropout: Dropout rate
        """
        super(LocationLayer, self).__init__()
        
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2, attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding, bias=False, stride=1,
            dilation=1, dropout=dropout
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim,
            bias=False, w_init_gain='tanh', dropout=dropout
        )

    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        """Process attention weights for location-aware attention."""
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    """
    Enhanced attention mechanism with modern improvements.
    """
    
    def __init__(self, 
                 attention_rnn_dim: int, 
                 embedding_dim: int, 
                 attention_dim: int,
                 attention_location_n_filters: int, 
                 attention_location_kernel_size: int,
                 dropout: float = 0.0,
                 temperature: float = 1.0):
        """
        Initialize Attention mechanism.
        
        Args:
            attention_rnn_dim: RNN dimension
            embedding_dim: Embedding dimension
            attention_dim: Attention dimension
            attention_location_n_filters: Number of location filters
            attention_location_kernel_size: Location kernel size
            dropout: Dropout rate
            temperature: Temperature for attention softmax
        """
        super(Attention, self).__init__()
        
        self.temperature = temperature
        
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim,
            bias=False, w_init_gain='tanh', dropout=dropout
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False,
            w_init_gain='tanh', dropout=dropout
        )
        self.v = LinearNorm(attention_dim, 1, bias=False, dropout=dropout)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim, dropout=dropout
        )
        
        self.score_mask_value = -float("inf")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def get_alignment_energies(self, 
                             query: Tensor, 
                             processed_memory: Tensor,
                             attention_weights_cat: Tensor) -> Tensor:
        """
        Compute alignment energies with temperature scaling.
        
        Args:
            query: Decoder output [B, H]
            processed_memory: Processed encoder outputs [B, T, H]
            attention_weights_cat: Attention weight history [B, 2, T]
            
        Returns:
            Alignment energies [B, T]
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))
        
        energies = energies.squeeze(-1) / self.temperature
        return energies

    def forward(self, 
                attention_hidden_state: Tensor, 
                memory: Tensor, 
                processed_memory: Tensor,
                attention_weights_cat: Tensor, 
                mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Forward pass of attention mechanism.
        
        Args:
            attention_hidden_state: Attention RNN output
            memory: Encoder outputs
            processed_memory: Processed encoder outputs
            attention_weights_cat: Previous attention weights
            mask: Padding mask
            
        Returns:
            Tuple of (attention_context, attention_weights)
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        if mask is not None:
            alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        
        # Apply dropout to attention weights during training
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)
        
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class ForwardAttentionV2(nn.Module):
    """
    Enhanced forward attention mechanism with modern improvements.
    """
    
    def __init__(self, 
                 attention_rnn_dim: int, 
                 embedding_dim: int, 
                 attention_dim: int,
                 attention_location_n_filters: int, 
                 attention_location_kernel_size: int,
                 dropout: float = 0.0,
                 temperature: float = 1.0):
        """Initialize ForwardAttentionV2 with enhanced features."""
        super(ForwardAttentionV2, self).__init__()
        
        self.temperature = temperature
        
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim,
            bias=False, w_init_gain='tanh', dropout=dropout
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False,
            w_init_gain='tanh', dropout=dropout
        )
        self.v = LinearNorm(attention_dim, 1, bias=False, dropout=dropout)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim, dropout=dropout
        )
        
        self.score_mask_value = -1e20  # More stable than -inf
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def get_alignment_energies(self, 
                             query: Tensor, 
                             processed_memory: Tensor,
                             attention_weights_cat: Tensor) -> Tensor:
        """Compute alignment energies with temperature scaling."""
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))
        
        energies = energies.squeeze(-1) / self.temperature
        return energies

    def forward(self, 
                attention_hidden_state: Tensor, 
                memory: Tensor, 
                processed_memory: Tensor,
                attention_weights_cat: Tensor, 
                mask: Optional[Tensor], 
                log_alpha: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with monotonic attention constraints.
        
        Returns:
            Tuple of (attention_context, attention_weights, log_alpha_new)
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        if mask is not None:
            log_energy = log_energy.masked_fill(mask, self.score_mask_value)

        # Compute forward attention with improved numerical stability
        log_alpha_shift_padded = []
        max_time = log_energy.size(1)
        
        for sft in range(2):
            if sft == 0:
                shifted = log_alpha
            else:
                shifte

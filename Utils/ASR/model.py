import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from .layers import MFCC, Attention, LinearNorm, ConvNorm, ConvBlock


class ASRCNN(nn.Module):
    """
    Enhanced ASR CNN model with modern PyTorch features.
    Supports both CTC and sequence-to-sequence training.
    """
    
    def __init__(self,
                 input_dim: int = 80,
                 hidden_dim: int = 256,
                 n_token: int = 35,
                 n_layers: int = 6,
                 token_embedding_dim: int = 256,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 use_spectral_norm: bool = False,
                 gradient_checkpointing: bool = False):
        """
        Initialize ASRCNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for CNN layers
            n_token: Number of output tokens/vocabulary size
            n_layers: Number of CNN layers
            token_embedding_dim: Token embedding dimension for S2S
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'swish')
            use_spectral_norm: Whether to use spectral normalization
            gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        """
        super().__init__()
        self.n_token = n_token
        self.n_down = 1
        self.hidden_dim = hidden_dim
        self.gradient_checkpointing = gradient_checkpointing
        
        # Enhanced MFCC processing
        self.to_mfcc = MFCC()
        
        # Initial convolution with modern initialization
        self.init_cnn = ConvNorm(
            input_dim // 2, 
            hidden_dim, 
            kernel_size=7, 
            padding=3, 
            stride=2
        )
        
        # Enhanced CNN layers with residual connections and modern normalization
        cnn_layers = []
        for i in range(n_layers):
            conv_block = ConvBlock(
                hidden_dim, 
                dropout=dropout,
                activation=activation,
                use_spectral_norm=use_spectral_norm
            )
            # Use LayerNorm instead of GroupNorm for better performance
            norm_layer = nn.LayerNorm(hidden_dim)
            cnn_layers.append(nn.Sequential(conv_block, norm_layer))
        
        self.cnns = nn.ModuleList(cnn_layers)
        
        # Enhanced projection with dropout
        self.projection = nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim // 2),
            nn.Dropout(dropout)
        )
        
        # CTC head with improved architecture
        self.ctc_linear = nn.Sequential(
            LinearNorm(hidden_dim // 2, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            LinearNorm(hidden_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            LinearNorm(hidden_dim, n_token)
        )
        
        # Enhanced S2S module
        self.asr_s2s = ASRS2S(
            embedding_dim=token_embedding_dim,
            hidden_dim=hidden_dim // 2,
            n_token=n_token,
            dropout=dropout,
            activation=activation
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize model weights with modern techniques."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, 
                x: Tensor, 
                src_key_padding_mask: Optional[Tensor] = None, 
                text_input: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            x: Input audio features [B, T, F]
            src_key_padding_mask: Padding mask for source
            text_input: Target text for S2S training [B, T]
            
        Returns:
            If text_input is None: CTC logits [B, T, V]
            Else: (CTC logits, S2S logits, S2S attention) tuple
        """
        # Feature extraction
        x = self.to_mfcc(x)
        x = self.init_cnn(x)
        
        # CNN processing with residual connections
        for i, cnn_layer in enumerate(self.cnns):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(cnn_layer, x, use_reentrant=False)
            else:
                residual = x
                x = cnn_layer(x)
                # Add residual connection every 2 layers
                if i % 2 == 1 and x.shape == residual.shape:
                    x = x + residual
        
        # Projection and preparation for output
        x = self.projection(x)
        x = x.transpose(1, 2).contiguous()  # [B, T, H]
        
        # CTC output
        ctc_logit = self.ctc_linear(x)
        
        # S2S output if text input provided
        if text_input is not None:
            _, s2s_logit, s2s_attn = self.asr_s2s(x, src_key_padding_mask, text_input)
            return ctc_logit, s2s_logit, s2s_attn
        else:
            return ctc_logit

    def get_feature(self, x: Tensor) -> Tensor:
        """Extract features without final classification layers."""
        x = self.to_mfcc(x.squeeze(1))
        x = self.init_cnn(x)
        
        for cnn_layer in self.cnns:
            x = cnn_layer(x)
        
        x = self.projection(x)
        return x

    def length_to_mask(self, lengths: Tensor) -> Tensor:
        """Convert lengths to boolean mask tensor."""
        max_len = lengths.max().item()
        batch_size = lengths.shape[0]
        
        # Create mask more efficiently
        mask = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype)
        mask = mask.unsqueeze(0).expand(batch_size, -1)
        mask = mask >= lengths.unsqueeze(1)
        
        return mask

    def get_future_mask(self, out_length: int, unmask_future_steps: int = 0) -> Tensor:
        """
        Generate causal mask for sequence generation.
        
        Args:
            out_length: Output sequence length
            unmask_future_steps: Number of future steps to unmask
            
        Returns:
            Boolean mask tensor [out_length, out_length]
        """
        # More efficient mask generation
        indices = torch.arange(out_length)
        mask = indices.unsqueeze(0) > (indices.unsqueeze(1) + unmask_future_steps)
        return mask


class ASRS2S(nn.Module):
    """
    Enhanced sequence-to-sequence decoder with modern PyTorch features.
    """
    
    def __init__(self,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 n_location_filters: int = 32,
                 location_kernel_size: int = 63,
                 n_token: int = 40,
                 dropout: float = 0.1,
                 activation: str = 'tanh',
                 use_attention_dropout: bool = True,
                 label_smoothing: float = 0.1):
        """
        Initialize ASRS2S decoder.
        
        Args:
            embedding_dim: Token embedding dimension
            hidden_dim: Hidden dimension
            n_location_filters: Number of location-aware attention filters
            location_kernel_size: Kernel size for location attention
            n_token: Vocabulary size
            dropout: Dropout rate
            activation: Activation function
            use_attention_dropout: Whether to use attention dropout
            label_smoothing: Label smoothing factor
        """
        super(ASRS2S, self).__init__()
        
        # Enhanced embedding with dropout
        self.embedding = nn.Embedding(n_token, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Improved initialization
        nn.init.normal_(self.embedding.weight, 0.0, embedding_dim ** -0.5)
        nn.init.zeros_(self.embedding.weight[0])  # padding token

        self.decoder_rnn_dim = hidden_dim
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        
        # Enhanced projection with residual connection
        self.project_to_n_symbols = nn.Sequential(
            nn.Linear(self.decoder_rnn_dim, self.decoder_rnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_rnn_dim, n_token)
        )
        
        # Enhanced attention with dropout
        self.attention_layer = Attention(
            self.decoder_rnn_dim,
            hidden_dim,
            hidden_dim,
            n_location_filters,
            location_kernel_size,
            dropout=dropout if use_attention_dropout else 0.0
        )
        
        # LSTM with improved initialization
        self.decoder_rnn = nn.LSTMCell(
            self.decoder_rnn_dim + embedding_dim, 
            self.decoder_rnn_dim
        )
        
        # Enhanced hidden projection
        self.project_to_hidden = nn.Sequential(
            LinearNorm(self.decoder_rnn_dim * 2, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout)
        )
        
        # Special tokens
        self.sos = 1
        self.eos = 2
        self.unk_index = 3
        self.pad_index = 0
        
        # Enhanced random masking with curriculum learning
        self.random_mask = 0.1
        self.register_buffer('training_step', torch.tensor(0))
        
        # Initialize LSTM weights
        self._initialize_lstm_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        return activations.get(activation.lower(), nn.Tanh())
    
    def _initialize_lstm_weights(self):
        """Initialize LSTM weights properly."""
        for name, param in self.decoder_rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)

    def initialize_decoder_states(self, memory: Tensor, mask: Optional[Tensor] = None):
        """
        Initialize decoder states with improved memory management.
        
        Args:
            memory: Encoder memory [B, L, H]
            mask: Memory mask [B, L]
        """
        B, L, H = memory.shape
        device = memory.device
        dtype = memory.dtype
        
        # Initialize states with proper device and dtype
        self.decoder_hidden = torch.zeros(B, self.decoder_rnn_dim, device=device, dtype=dtype)
        self.decoder_cell = torch.zeros(B, self.decoder_rnn_dim, device=device, dtype=dtype)
        self.attention_weights = torch.zeros(B, L, device=device, dtype=dtype)
        self.attention_weights_cum = torch.zeros(B, L, device=device, dtype=dtype)
        self.attention_context = torch.zeros(B, H, device=device, dtype=dtype)
        
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        
        # Curriculum learning for random masking
        if self.training:
            progress = min(self.training_step.item() / 10000, 1.0)
            self.current_random_mask = self.random_mask * (1.0 - progress * 0.5)
        else:
            self.current_random_mask = 0.0

    def forward(self, 
                memory: Tensor, 
                memory_mask: Optional[Tensor], 
                text_input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the S2S decoder.
        
        Args:
            memory: Encoder memory [B, L, H]
            memory_mask: Memory padding mask [B, L]
            text_input: Target text [B, T]
            
        Returns:
            Tuple of (hidden_outputs, logit_outputs, alignments)
        """
        self.initialize_decoder_states(memory, memory_mask)
        
        # Enhanced random masking with curriculum learning
        if self.training and self.current_random_mask > 0:
            random_mask = torch.rand_like(text_input, dtype=torch.float32) < self.current_random_mask
            _text_input = text_input.clone()
            _text_input.masked_fill_(random_mask, self.unk_index)
        else:
            _text_input = text_input
        
        # Embedding with dropout
        decoder_inputs = self.embedding(_text_input).transpose(0, 1)  # [T, B, E]
        decoder_inputs = self.embedding_dropout(decoder_inputs)
        
        # Start token embedding
        start_embedding = self.embedding(
            torch.full((decoder_inputs.size(1),), self.sos, 
                      device=decoder_inputs.device, dtype=torch.long)
        )
        start_embedding = self.embedding_dropout(start_embedding)
        decoder_inputs = torch.cat([start_embedding.unsqueeze(0), decoder_inputs], dim=0)

        # Decoding loop with enhanced tracking
        hidden_outputs, logit_outputs, alignments = [], [], []
        max_steps = decoder_inputs.size(0)
        
        for step in range(max_steps):
            decoder_input = decoder_inputs[step]
            hidden, logit, attention_weights = self.decode(decoder_input)
            
            hidden_outputs.append(hidden)
            logit_outputs.append(logit)
            alignments.append(attention_weights)

        # Parse outputs
        hidden_outputs, logit_outputs, alignments = self.parse_decoder_outputs(
            hidden_outputs, logit_outputs, alignments)
        
        # Update training step
        if self.training:
            self.training_step += 1

        return hidden_outputs, logit_outputs, alignments

    def decode(self, decoder_input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Single decoding step with enhanced processing.
        
        Args:
            decoder_input: Current decoder input [B, E]
            
        Returns:
            Tuple of (hidden, logit, attention_weights)
        """
        # LSTM cell forward pass
        cell_input = torch.cat([decoder_input, self.attention_context], dim=-1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            cell_input, (self.decoder_hidden, self.decoder_cell))

        # Attention computation with enhanced context
        attention_weights_cat = torch.stack([
            self.attention_weights,
            self.attention_weights_cum
        ], dim=1)  # [B, 2, L]

        self.attention_context, self.attention_weights = self.attention_layer(
            self.decoder_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask)

        # Update cumulative attention
        self.attention_weights_cum = self.attention_weights_cum + self.attention_weights

        # Hidden state computation
        hidden_and_context = torch.cat([self.decoder_hidden, self.attention_context], dim=-1)
        hidden = self.project_to_hidden(hidden_and_context)

        # Output projection with enhanced dropout
        logit = self.project_to_n_symbols(F.dropout(hidden, self.dropout, self.training))

        return hidden, logit, self.attention_weights

    def parse_decoder_outputs(self, 
                            hidden: list, 
                            logit: list, 
                            alignments: list) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parse decoder outputs into proper tensor format.
        
        Args:
            hidden: List of hidden states
            logit: List of logits
            alignments: List of attention weights
            
        Returns:
            Tuple of stacked tensors
        """
        # Stack and transpose efficiently
        alignments = torch.stack(alignments, dim=1)  # [B, T+1, L]
        logit = torch.stack(logit, dim=1)  # [B, T+1, V]
        hidden = torch.stack(hidden, dim=1)  # [B, T+1, H]

        return hidden, logit, alignments

    @torch.jit.export
    def generate(self, 
                 memory: Tensor, 
                 memory_mask: Optional[Tensor] = None,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.9) -> Tensor:
        """
        Generate sequences using various decoding strategies.
        
        Args:
            memory: Encoder memory [B, L, H]
            memory_mask: Memory mask [B, L]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated sequences [B, T]
        """
        self.eval()
        self.initialize_decoder_states(memory, memory_mask)
        
        batch_size = memory.size(0)
        device = memory.device
        
        # Initialize with SOS token
        generated = torch.full((batch_size, 1), self.sos, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            if finished.all():
                break
                
            # Get current input
            current_input = self.embedding(generated[:, -1])
            current_input = self.embedding_dropout(current_input)
            
            # Decode step
            _, logit, _ = self.decode(current_input)
            
            # Apply temperature and sampling
            logit = logit / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
                logit[indices_to_remove] = float('-inf')
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logit, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logit[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logit, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Update finished sequences
            finished = finished | (next_token.squeeze(1) == self.eos)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated

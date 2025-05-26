import os
import os.path as osp
import copy
import math
from typing import Optional, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm # Import for use in CNNs
from munch import Munch # For flexible configuration loading
import yaml # For loading YAML configuration files

# --- Helper Layers ---

class LinearNorm(torch.nn.Module):
    """
    A linear layer with Xavier uniform initialization.

    This layer is a standard fully connected neural network layer
    with a specific weight initialization strategy known as Xavier uniform.
    Xavier initialization helps in keeping the signal variance constant
    across layers, which can prevent vanishing/exploding gradients during training.
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = 'linear'):
        """
        Initializes the LinearNorm layer.

        Args:
            in_dim (int): The number of input features.
            out_dim (int): The number of output features.
            bias (bool): Whether to include a bias term in the linear layer. Defaults to True.
            w_init_gain (str): The gain value to use for Xavier uniform initialization.
                               'linear' is typical for linear activations, 'relu' for ReLU.
        """
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the LinearNorm layer.

        Args:
            x (torch.Tensor): The input tensor to the linear layer.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.
        """
        return self.linear_layer(x)

class LayerNorm(nn.Module):
    """
    Layer Normalization module designed for channel-first tensors (Batch, Channels, Time).

    This module applies Layer Normalization across the feature dimension for inputs
    that are typically in a (B, C, T) format, common in convolutional neural networks
    processing sequential data like spectrograms or text features. It internally
    transposes the tensor to (B, T, C) for standard PyTorch `F.layer_norm` application,
    then transposes it back to (B, C, T).
    """
    def __init__(self, channels: int, eps: float = 1e-5):
        """
        Initializes the LayerNorm module.

        Args:
            channels (int): The number of channels (features) in the input tensor.
            eps (float): A small value added to the denominator for numerical stability. Defaults to 1e-5.
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the LayerNorm module.

        Args:
            x (torch.Tensor): The input tensor of shape (Batch, Channels, Time).

        Returns:
            torch.Tensor: The layer-normalized output tensor of shape (Batch, Channels, Time).
        """
        # Transpose to (Batch, Time, Channels) for standard LayerNorm application
        x = x.transpose(1, -1)
        # Apply layer normalization
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        # Transpose back to (Batch, Channels, Time)
        return x.transpose(1, -1)

# --- Core Model Components ---

class KokoroTextEncoder(nn.Module):
    """
    The Text Encoder component of the Kokoro-82M model.

    This module converts a sequence of text tokens (e.g., phoneme IDs) into
    a rich sequence of hidden text features. It employs a combination of
    an embedding layer, a stack of 1D Convolutional Neural Networks (CNNs)
    for local context extraction, and a bidirectional LSTM for capturing
    long-range dependencies.
    """
    def __init__(self, channels: int = 256, kernel_size: int = 5, depth: int = 4,
                 n_symbols: int = 256, dropout: float = 0.1):
        """
        Initializes the KokoroTextEncoder.

        Args:
            channels (int): The dimensionality of the text features (embedding size and CNN channels). Defaults to 256.
            kernel_size (int): The kernel size for the 1D CNN layers. Defaults to 5.
            depth (int): The number of CNN layers in the encoder. Defaults to 4.
            n_symbols (int): The total number of unique input symbols (e.g., phonemes + special tokens). Defaults to 256.
            dropout (float): The dropout rate applied after ReLU in CNNs. Defaults to 0.1.
        """
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        
        padding = (kernel_size - 1) // 2 # Ensures output sequence length matches input
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels), # Apply LayerNorm after convolution
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ))

        # Bidirectional LSTM to capture contextual information
        # Output dim is `channels` (channels//2 * 2 for bidirectional)
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(channels, channels) # Final projection to expected feature dimension

    def forward(self, x: torch.Tensor, input_lengths: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of the Text Encoder.

        Args:
            x (torch.Tensor): Input tensor of text token IDs, shape `(Batch, MaxTextSequenceLength)`.
            input_lengths (Optional[torch.Tensor]): Lengths of each text sequence in the batch,
                                                    shape `(Batch,)`. Used for packing padded sequences.
            mask (Optional[torch.Tensor]): Boolean mask indicating valid (True) vs. padded (False)
                                            elements, shape `(Batch, MaxTextSequenceLength)`.
                                            Used to zero out padded regions after CNNs.

        Returns:
            torch.Tensor: Encoded text features, shape `(Batch, MaxTextSequenceLength, Channels)`.
        """
        x = self.embedding(x)  # (B, T_text, channels)
        x = x.transpose(1, 2)  # (B, channels, T_text) - suitable for Conv1d

        # Apply mask to zero out padded values if provided
        if mask is not None:
            # Expand mask to match CNN input dimensions (B, 1, T_text)
            mask_expanded = mask.to(x.device).unsqueeze(1)
            x.masked_fill_(mask_expanded, 0.0)
        
        # Pass through CNN layers
        for c in self.cnn:
            x = c(x)
            if mask is not None: # Re-apply mask after each CNN layer
                x.masked_fill_(mask_expanded, 0.0)
                
        x = x.transpose(1, 2)  # (B, T_text, channels) - suitable for LSTM

        # Handle padded sequences for LSTM
        if input_lengths is not None:
            # pack_padded_sequence requires lengths on CPU
            input_lengths_cpu = input_lengths.cpu() 
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths_cpu, batch_first=True, enforce_sorted=False
            )

        self.lstm.flatten_parameters() # Optimize LSTM for faster execution
        x, _ = self.lstm(x) # x is (PackedSequence) or (B, T_text, channels)
        
        # Unpack sequence if it was packed
        if input_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        # Final linear projection
        x = self.output_proj(x)
        return x

class VoicePacketEmbedding(nn.Module):
    """
    Generates a combined voice and language embedding for multi-speaker and multi-lingual synthesis.

    This module acts as the "speaker" or "style" conditioning mechanism. Instead of a
    complex style-diffusion or reference encoder as in original StyleTTS-2, Kokoro-82M
    uses a simpler approach of pre-trained, discrete voice and language embeddings.
    """
    def __init__(self, num_voices: int = 54, voice_dim: int = 256, num_languages: int = 8):
        """
        Initializes the VoicePacketEmbedding.

        Args:
            num_voices (int): The total number of unique voices supported by the model. Defaults to 54.
            voice_dim (int): The dimensionality of the voice embeddings. Defaults to 256.
            num_languages (int): The total number of unique languages supported. Defaults to 8.
        """
        super().__init__()
        self.num_voices = num_voices
        self.voice_dim = voice_dim
        # Embeddings for individual voices
        self.voice_embeddings = nn.Embedding(num_voices, voice_dim)
        # Embeddings for languages (a quarter of voice_dim, as seen in common multi-lingual setups)
        self.language_embeddings = nn.Embedding(num_languages, voice_dim // 4)
        # Linear projection to combine voice and language embeddings into a single vector
        self.proj = nn.Linear(voice_dim + voice_dim // 4, voice_dim)
        
    def forward(self, voice_id: torch.Tensor, language_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass to generate the combined voice/language embedding.

        Args:
            voice_id (torch.Tensor): A tensor of voice IDs. Can be a scalar (0-dim),
                                     a 1D tensor `(Batch,)`, or a 2D tensor `(Batch, 1)`.
            language_id (Optional[torch.Tensor]): An optional tensor of language IDs,
                                                  with similar dimensionality rules as `voice_id`.

        Returns:
            torch.Tensor: The combined voice and language embedding, shape `(Batch, VoiceDim)`
                          or `(VoiceDim,)` if input was scalar.
        """
        # Get voice embedding. Handles scalar input by adding a batch dimension.
        voice_emb = self.voice_embeddings(voice_id) # (B, voice_dim) or (voice_dim,)

        if language_id is not None:
            # Get language embedding. Handles scalar input by adding a batch dimension.
            lang_emb = self.language_embeddings(language_id) # (B, voice_dim // 4) or (voice_dim // 4,)

            # Ensure both embeddings have a consistent batch dimension for concatenation
            if voice_emb.dim() == 1 and lang_emb.dim() == 1:
                # Both are scalar outputs from embedding, unsqueeze to (1, D) for concat
                voice_emb = voice_emb.unsqueeze(0)
                lang_emb = lang_emb.unsqueeze(0)
            elif voice_emb.dim() == 1 and lang_emb.dim() == 2:
                # Voice_emb is scalar, lang_emb is batched. Unsqueeze voice_emb to (1, D)
                # and then expand to match batch size of lang_emb if needed (not here, concat handles)
                voice_emb = voice_emb.unsqueeze(0)

            # Concatenate voice and language embeddings
            combined = torch.cat([voice_emb, lang_emb], dim=-1)
            # Project to the final voice_dim
            return self.proj(combined)
            
        return voice_emb

class KokoroProsodyPredictor(nn.Module):
    """
    Predicts prosodic features (F0, energy, and duration) from text features and voice embedding.

    This module is crucial for controlling the expressiveness and naturalness of the synthesized speech.
    It takes context from both the linguistic content (text features) and the speaker/language identity.
    """
    def __init__(self, text_dim: int = 256, voice_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        """
        Initializes the KokoroProsodyPredictor.

        Args:
            text_dim (int): The dimensionality of the input text features. Defaults to 256.
            voice_dim (int): The dimensionality of the input voice embedding. Defaults to 256.
            hidden_dim (int): The hidden dimensionality used within the predictor network.
                              Note: This is used for the internal layers, while the output
                              features might have a different effective dimension (e.g., prosody_dim in config).
                              Here, `hidden_dim` is effectively `prosody_dim * 2` from the overall config.
            dropout (float): Dropout rate applied in the prosody prediction network. Defaults to 0.1.
        """
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.voice_proj = nn.Linear(voice_dim, hidden_dim)
        
        # A simple feed-forward network to predict prosody
        self.prosody_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Concatenates text_proj and voice_proj
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Separate heads for F0, energy, and duration
        self.f0_head = nn.Linear(hidden_dim // 2, 1) # Predicts a single scalar F0 value per text token
        self.energy_head = nn.Linear(hidden_dim // 2, 1) # Predicts a single scalar energy value per text token
        self.duration_head = nn.Linear(hidden_dim // 2, 1) # Predicts a single scalar duration value per text token
        
    def forward(self, text_features: torch.Tensor, voice_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass of the Prosody Predictor.

        Args:
            text_features (torch.Tensor): Text features from the `KokoroTextEncoder`,
                                          shape `(Batch, MaxTextSequenceLength, TextDim)`.
            voice_embedding (torch.Tensor): Combined voice/language embedding from
                                            `VoicePacketEmbedding`, shape `(Batch, VoiceDim)` or `(VoiceDim,)`.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'f0': Predicted F0 values, shape `(Batch, MaxTextSequenceLength)`.
                - 'energy': Predicted energy values, shape `(Batch, MaxTextSequenceLength)`.
                - 'duration': Predicted duration values (0-1), shape `(Batch, MaxTextSequenceLength)`.
                - 'prosody_features': Intermediate prosody features, shape `(Batch, MaxTextSequenceLength, HiddenDim // 2)`.
        """
        text_proj = self.text_proj(text_features) # (B, T_text, HiddenDim)
        voice_proj = self.voice_proj(voice_embedding) # (B, HiddenDim) or (HiddenDim,)
        
        # Expand voice embedding to match the sequence length of text features
        if voice_proj.dim() == 0: # If voice_embedding was a scalar (e.g., single int voice_id)
            voice_proj = voice_proj.unsqueeze(0) # Make it (1,)
        if voice_proj.dim() == 1: # If voice_proj is (D,), make it (1, D) for broadcasting
            voice_proj = voice_proj.unsqueeze(0)
        
        # Expand voice_proj to (Batch, MaxTextSequenceLength, HiddenDim)
        # This effectively tiles the voice embedding for each text token
        voice_proj_expanded = voice_proj.unsqueeze(1).expand(-1, text_proj.size(1), -1)
            
        # Concatenate text features and expanded voice features
        combined = torch.cat([text_proj, voice_proj_expanded], dim=-1) # (B, T_text, HiddenDim * 2)
        
        # Pass through the prosody prediction network
        prosody_features = self.prosody_net(combined) # (B, T_text, HiddenDim // 2)
        
        # Predict F0, energy, and duration using separate linear heads
        # Squeeze the last dimension to remove the singleton dimension (e.g., from (..., 1) to (...))
        f0 = self.f0_head(prosody_features).squeeze(-1)
        energy = self.energy_head(prosody_features).squeeze(-1)
        # Duration prediction typically uses sigmoid to constrain values between 0 and 1
        duration = torch.sigmoid(self.duration_head(prosody_features)).squeeze(-1)
        
        return {
            'f0': f0,
            'energy': energy,
            'duration': duration,
            'prosody_features': prosody_features # Return for use by the Decoder
        }

class KokoroDecoder(nn.Module):
    """
    The main decoder responsible for generating mel-spectrograms from
    text features, voice embeddings, and predicted prosody features.

    This module forms the core of the acoustic model, taking all available
    linguistic and prosodic information to produce the target speech representation.
    It's a "decoder-only" architecture, simplifying from StyleTTS-2's diffusion model.
    """
    def __init__(self, 
                 text_dim: int = 256, 
                 voice_dim: int = 256, 
                 prosody_dim: int = 128, # This should match the output dim of prosody_net (hidden_dim // 2)
                 mel_dim: int = 80, 
                 hidden_dim: int = 512, # Internal hidden dimension for decoder layers
                 num_layers: int = 6, # Number of decoder layers
                 dropout: float = 0.1):
        """
        Initializes the KokoroDecoder.

        Args:
            text_dim (int): Dimensionality of text features. Defaults to 256.
            voice_dim (int): Dimensionality of voice embedding. Defaults to 256.
            prosody_dim (int): Dimensionality of prosody features. Defaults to 128.
            mel_dim (int): Dimensionality of output mel-spectrograms. Defaults to 80.
            hidden_dim (int): Hidden dimension for self-attention and FFN in decoder layers. Defaults to 512.
            num_layers (int): Number of decoder layers. Defaults to 6.
            dropout (float): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        
        # Input projection layer to combine all input features
        self.input_proj = nn.Linear(text_dim + voice_dim + prosody_dim, hidden_dim)
        
        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            KokoroDecoderLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Output projection to mel_dim, with an intermediate ReLU and dropout
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, mel_dim)
        )
        
        # Postnet for residual mel-spectrogram refinement
        self.postnet = KokoroPostnet(mel_dim, hidden_dim // 4)
        
    def forward(self, text_features: torch.Tensor, voice_embedding: torch.Tensor, 
                prosody_features: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass of the Decoder.

        Args:
            text_features (torch.Tensor): Text features from `KokoroTextEncoder`,
                                          shape `(Batch, MaxSeqLen, TextDim)`.
            voice_embedding (torch.Tensor): Combined voice/language embedding from
                                            `VoicePacketEmbedding`, shape `(Batch, VoiceDim)` or `(VoiceDim,)`.
            prosody_features (torch.Tensor): Prosody features from `KokoroProsodyPredictor`,
                                             shape `(Batch, MaxSeqLen, ProsodyDim)`.
            lengths (Optional[torch.Tensor]): Lengths of the sequences in the batch, used for masking.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'mel_before': Mel-spectrograms before Postnet, shape `(Batch, MaxSeqLen, MelDim)`.
                - 'mel_after': Mel-spectrograms after Postnet refinement, shape `(Batch, MaxSeqLen, MelDim)`.
        """
        batch_size, seq_len, _ = text_features.shape
        
        # Expand voice embedding to match the sequence length, similar to ProsodyPredictor
        if voice_embedding.dim() == 0:
            voice_embedding = voice_embedding.unsqueeze(0)
        if voice_embedding.dim() == 1:
            voice_embedding = voice_embedding.unsqueeze(0)
        voice_embedding_expanded = voice_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            
        # Concatenate all input features for the decoder
        combined_features = torch.cat([text_features, voice_embedding_expanded, prosody_features], dim=-1)
        
        # Project combined features to the decoder's hidden dimension
        x = self.input_proj(combined_features)
        
        # Pass through stack of decoder layers
        for layer in self.decoder_layers:
            x = layer(x, lengths)
            
        # Project to mel-spectrogram dimension (before Postnet)
        mel_before = self.output_proj(x)
        
        # Apply Postnet for residual refinement.
        # The Postnet output is added to the mel_before to get the final mel_after.
        mel_after = self.postnet(mel_before) + mel_before
        
        return {
            'mel_before': mel_before,
            'mel_after': mel_after
        }

class KokoroDecoderLayer(nn.Module):
    """
    A single layer of the Kokoro Decoder, similar to a Transformer decoder layer.

    This layer incorporates self-attention to capture dependencies within the mel-spectrogram
    sequence and a feed-forward network for feature transformation. It also uses
    Layer Normalization for stability and dropout for regularization.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1, num_heads: int = 8):
        """
        Initializes a single KokoroDecoderLayer.

        Args:
            hidden_dim (int): The dimensionality of features processed by this layer.
            dropout (float): Dropout rate. Defaults to 0.1.
            num_heads (int): Number of attention heads for multi-head self-attention. Defaults to 8.
        """
        super().__init__()
        # Multi-head self-attention module
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim) # LayerNorm after attention
        
        # Position-wise Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), # Expand to a larger dimension
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), # Project back to original dimension
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim) # LayerNorm after FFN
        self.dropout_layer = nn.Dropout(dropout) # Additional dropout for residuals
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Performs the forward pass of a single Decoder Layer.

        Args:
            x (torch.Tensor): Input tensor to the layer, shape `(Batch, SequenceLength, HiddenDim)`.
            lengths (Optional[torch.Tensor]): Lengths of the sequences, used to create an attention mask.

        Returns:
            torch.Tensor: Output tensor of the layer, shape `(Batch, SequenceLength, HiddenDim)`.
        """
        attn_mask = None
        if lengths is not None:
            # Create a boolean mask where True indicates padded positions to be ignored by attention
            max_len = x.size(1)
            attn_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            
        # Self-Attention Block
        residual = x # Store input for residual connection
        x_attn, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        # Add residual and apply Layer Normalization
        x = self.norm1(residual + self.dropout_layer(x_attn))
        
        # Feed-Forward Network Block
        residual = x # Store input for residual connection
        x_ffn = self.ffn(x)
        # Add residual and apply Layer Normalization
        x = self.norm2(residual + x_ffn)
        
        return x

class KokoroPostnet(nn.Module):
    """
    The Postnet module for refining mel-spectrograms.

    The Postnet is a convolutional network that takes the mel-spectrogram output
    from the main decoder and predicts a residual correction. This residual is then
    added to the original mel-spectrogram, allowing for fine-grained improvements
    in spectral detail and quality.
    """
    def __init__(self, mel_dim: int = 80, hidden_dim: int = 128, num_layers: int = 5, kernel_size: int = 5):
        """
        Initializes the KokoroPostnet.

        Args:
            mel_dim (int): The dimensionality of the input and output mel-spectrograms. Defaults to 80.
            hidden_dim (int): The number of channels in the hidden convolutional layers. Defaults to 128.
            num_layers (int): The number of convolutional layers in the Postnet. Defaults to 5.
            kernel_size (int): The kernel size for the 1D convolutional layers. Defaults to 5.
        """
        super().__init__()
        
        padding = (kernel_size - 1) // 2 # Padding to preserve sequence length
        self.convs = nn.ModuleList()
        
        # First convolutional layer: mel_dim -> hidden_dim
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(mel_dim, hidden_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(), # Tanh activation
                nn.Dropout(0.5)
            )
        )
        
        # Intermediate convolutional layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2): # -2 for first and last layers
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5)
                )
            )
            
        # Last convolutional layer: hidden_dim -> mel_dim
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(hidden_dim, mel_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(0.5) # No activation on the final layer (predicts residual)
            )
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Postnet.

        Args:
            x (torch.Tensor): Input mel-spectrograms from the Decoder,
                              shape `(Batch, SequenceLength, MelDim)`.

        Returns:
            torch.Tensor: Predicted residual for mel-spectrogram refinement,
                          shape `(Batch, SequenceLength, MelDim)`.
        """
        # Transpose to (Batch, MelDim, SequenceLength) for Conv1d
        x = x.transpose(1, 2)
        
        # Pass through all convolutional layers
        for conv in self.convs:
            x = conv(x)
            
        # Transpose back to (Batch, SequenceLength, MelDim)
        x = x.transpose(1, 2)
        return x

# --- Full KokoroTTS Model ---

class KokoroTTS(nn.Module):
    """
    The complete Kokoro-82M Text-to-Speech (TTS) model.

    This model integrates the Text Encoder, Voice Packet Embedding, Prosody Predictor,
    and Mel-Spectrogram Decoder with Postnet to perform end-to-end text-to-mel synthesis.
    It does NOT include the vocoder (e.g., ISTFTNet), which is an external component
    required to convert the mel-spectrograms to audible waveforms.
    """
    def __init__(self, 
                 n_symbols: int = 256,
                 num_voices: int = 54,
                 num_languages: int = 8,
                 text_dim: int = 256,
                 voice_dim: int = 256,
                 prosody_dim: int = 128, # Output dim of prosody features
                 mel_dim: int = 80,
                 hidden_dim: int = 512, # Decoder's internal hidden dim
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1):
        """
        Initializes the KokoroTTS model.

        Args:
            n_symbols (int): Number of unique input text symbols (e.g., phonemes). Defaults to 256.
            num_voices (int): Number of distinct voice IDs. Defaults to 54.
            num_languages (int): Number of distinct language IDs. Defaults to 8.
            text_dim (int): Dimensionality of text features. Defaults to 256.
            voice_dim (int): Dimensionality of voice embeddings. Defaults to 256.
            prosody_dim (int): Dimensionality of predicted prosody features. Defaults to 128.
            mel_dim (int): Dimensionality of output mel-spectrograms. Defaults to 80.
            hidden_dim (int): Internal hidden dimension for the decoder. Defaults to 512.
            num_decoder_layers (int): Number of layers in the mel-spectrogram decoder. Defaults to 6.
            dropout (float): Dropout rate used across various modules. Defaults to 0.1.
        """
        super().__init__()
        
        self.text_encoder = KokoroTextEncoder(
            channels=text_dim, 
            n_symbols=n_symbols,
            dropout=dropout
        )
        
        self.voice_embedding = VoicePacketEmbedding(
            num_voices=num_voices,
            voice_dim=voice_dim,
            num_languages=num_languages
        )
        
        # Prosody predictor's hidden_dim is typically twice the prosody_dim
        # because it concatenates text_proj and voice_proj each of which are
        # projected to `hidden_dim` (which is `prosody_dim * 2`).
        self.prosody_predictor = KokoroProsodyPredictor(
            text_dim=text_dim,
            voice_dim=voice_dim,
            hidden_dim=prosody_dim * 2, # Internal hidden_dim for predictor, matching output expected concatenated input
            dropout=dropout
        )
        
        self.decoder = KokoroDecoder(
            text_dim=text_dim,
            voice_dim=voice_dim,
            prosody_dim=prosody_dim, # The output dimension of prosody_features used by decoder
            mel_dim=mel_dim,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
    def forward(self, text_tokens: torch.Tensor, voice_ids: torch.Tensor,
                language_ids: Optional[torch.Tensor] = None,
                text_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass of the KokoroTTS model for training.

        Args:
            text_tokens (torch.Tensor): Input text token IDs, shape `(Batch, MaxTextSequenceLength)`.
            voice_ids (torch.Tensor): Voice IDs for each item in the batch, shape `(Batch,)`.
            language_ids (Optional[torch.Tensor]): Language IDs for each item in the batch, shape `(Batch,)`.
            text_lengths (Optional[torch.Tensor]): Actual lengths of text sequences, shape `(Batch,)`.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing model outputs:
                - 'mel_before': Predicted mel-spectrograms before Postnet, shape `(B, T_mel, MelDim)`.
                - 'mel_after': Predicted mel-spectrograms after Postnet, shape `(B, T_mel, MelDim)`.
                - 'f0': Predicted fundamental frequency, shape `(B, T_text)`.
                - 'energy': Predicted energy, shape `(B, T_text)`.
                - 'duration': Predicted duration, shape `(B, T_text)`.
        """
        # 1. Encode text tokens
        text_features = self.text_encoder(text_tokens, text_lengths) # (B, T_text, TextDim)

        # 2. Generate voice/language embeddings
        voice_embeddings = self.voice_embedding(voice_ids, language_ids) # (B, VoiceDim)

        # 3. Predict prosody (F0, energy, duration) and get prosody features
        prosody_output = self.prosody_predictor(text_features, voice_embeddings)
        # prosody_output contains 'f0', 'energy', 'duration', 'prosody_features'

        # 4. Decode mel-spectrograms using all features
        decoder_output = self.decoder(
            text_features, 
            voice_embeddings, # Note: voice_embedding will be expanded within the decoder
            prosody_output['prosody_features'], # Use the intermediate prosody features
            text_lengths # Use text_lengths for masking in decoder
        )
        # decoder_output contains 'mel_before', 'mel_after'
        
        return {
            'mel_before': decoder_output['mel_before'],
            'mel_after': decoder_output['mel_after'],
            'f0': prosody_output['f0'],
            'energy': prosody_output['energy'],
            'duration': prosody_output['duration']
        }
    
    def inference(self, text_tokens: torch.Tensor, voice_id: Union[int, torch.Tensor],
                  language_id: Optional[Union[int, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Performs inference (text-to-mel synthesis) for a single utterance.

        This method is designed for single-item inference. It handles scalar voice_id/language_id
        and automatically adds a batch dimension to inputs if needed.

        Args:
            text_tokens (torch.Tensor): Input text token IDs for a single utterance.
                                        Can be `(SequenceLength,)` or `(1, SequenceLength)`.
            voice_id (Union[int, torch.Tensor]): The voice ID for synthesis. Can be an integer
                                                 or a 0-dim/1-dim tensor.
            language_id (Optional[Union[int, torch.Tensor]]): The language ID for synthesis.
                                                               Can be an integer or a 0-dim/1-dim tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing model outputs for inference:
                - 'mel_before': Predicted mel-spectrograms before Postnet, shape `(1, T_mel, MelDim)`.
                - 'mel_after': Predicted mel-spectrograms after Postnet, shape `(1, T_mel, MelDim)`.
                - 'f0': Predicted fundamental frequency, shape `(1, T_text)`.
                - 'energy': Predicted energy, shape `(1, T_text)`.
                - 'duration': Predicted duration, shape `(1, T_text)`.
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculations for inference
            device = text_tokens.device # Ensure all tensors are on the same device

            # Ensure voice_id is a 1-element tensor on the correct device
            if isinstance(voice_id, int):
                voice_id_tensor = torch.tensor([voice_id], device=device)
            elif voice_id.dim() == 0:
                voice_id_tensor = voice_id.unsqueeze(0).to(device)
            else:
                voice_id_tensor = voice_id.to(device)

            # Ensure language_id is a 1-element tensor, if provided, on the correct device
            language_id_tensor = None
            if language_id is not None:
                if isinstance(language_id, int):
                    language_id_tensor = torch.tensor([language_id], device=device)
                elif language_id.dim() == 0:
                    language_id_tensor = language_id.unsqueeze(0).to(device)
                else:
                    language_id_tensor = language_id.to(device)
                
            # Ensure text_tokens has a batch dimension (e.g., from (T,) to (1, T))
            input_text_tokens = text_tokens.unsqueeze(0) if text_tokens.dim() == 1 else text_tokens

            # Call the main forward pass with the prepared tensors
            # For inference, text_lengths are often implicitly handled by attention masks
            # if sequence is not too long, or can be passed as input_text_tokens.shape[1]
            # if explicit padding is used. Here, we pass None as it's typically handled by attention mechanism.
            return self.forward(
                input_text_tokens,
                voice_id_tensor,
                language_id_tensor,
                text_lengths=None # For inference, masking handles variable lengths
            )

# --- Model Building and Utility Functions ---

def build_kokoro_model(config: Dict) -> KokoroTTS:
    """
    Builds an instance of the KokoroTTS model based on a configuration dictionary.

    Args:
        config (Dict): A dictionary containing model configuration parameters.
                       Expected keys: 'n_symbols', 'num_voices', 'num_languages',
                       'text_dim', 'voice_dim', 'prosody_dim', 'mel_dim',
                       'hidden_dim', 'num_decoder_layers', 'dropout'.

    Returns:
        KokoroTTS: An initialized KokoroTTS model.
    """
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
    """
    Combined loss function for training the Kokoro-82M TTS model.

    This loss function typically comprises:
    - L1 loss for mel-spectrogram reconstruction (before and after Postnet).
    - MSE loss for prosodic features (F0, energy, duration).
    """
    def __init__(self, mel_weight: float = 1.0, prosody_weight: float = 0.1):
        """
        Initializes the KokoroLoss module.

        Args:
            mel_weight (float): Weight for the mel-spectrogram reconstruction loss. Defaults to 1.0.
            prosody_weight (float): Weight for the prosody prediction losses. Defaults to 0.1.
        """
        super().__init__()
        self.mel_weight = mel_weight
        self.prosody_weight = prosody_weight
        self.l1_loss = nn.L1Loss() # For mel-spectrograms (often L1 is preferred for stability)
        self.mse_loss = nn.MSELoss() # For continuous values like F0 and energy, and duration

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculates the total loss for a given set of predictions and target values.

        Args:
            predictions (Dict[str, torch.Tensor]): A dictionary containing model outputs
                                                   ('mel_before', 'mel_after', 'f0', 'energy', 'duration').
            targets (Dict[str, torch.Tensor]): A dictionary containing ground truth targets
                                               ('mel', 'f0', 'energy', 'duration').

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'total_loss': The combined loss.
                - 'mel_loss': The mel-spectrogram reconstruction loss.
                - 'prosody_loss': The total prosody prediction loss.
        """
        # Mel-spectrogram losses (before and after Postnet)
        mel_loss_before = self.l1_loss(predictions['mel_before'], targets['mel'])
        mel_loss_after = self.l1_loss(predictions['mel_after'], targets['mel'])
        mel_loss = mel_loss_before + mel_loss_after
        
        # Prosody losses (F0, energy, duration)
        prosody_loss = torch.tensor(0.0, device=predictions['f0'].device) # Initialize on correct device
        
        # Only add loss if the target exists and shapes match
        if 'f0' in targets and predictions['f0'].shape == targets['f0'].shape:
            prosody_loss += self.mse_loss(predictions['f0'], targets['f0'])
        if 'energy' in targets and predictions['energy'].shape == targets['energy'].shape:
            prosody_loss += self.mse_loss(predictions['energy'], targets['energy'])
        if 'duration' in targets and predictions['duration'].shape == targets['duration'].shape:
            prosody_loss += self.mse_loss(predictions['duration'], targets['duration'])
            
        # Total weighted loss
        total_loss = self.mel_weight * mel_loss + self.prosody_weight * prosody_loss
        
        return {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'prosody_loss': prosody_loss
        }

def load_kokoro_checkpoint(model: nn.Module, checkpoint_path: str, device: str = 'cpu') -> Tuple[int, int]:
    """
    Loads a model checkpoint from a specified path.

    Args:
        model (nn.Module): The model instance to load the state dictionary into.
        checkpoint_path (str): The file path to the checkpoint.
        device (str): The device to load the checkpoint onto ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        Tuple[int, int]: A tuple containing the loaded epoch and step.
                         Returns (0, 0) if epoch/step info is not in checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    
    # Load checkpoint, mapping to specified device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dictionary, ensuring strict matching
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Return epoch and step if available, otherwise default to 0
    return checkpoint.get('epoch', 0), checkpoint.get('step', 0)

def save_kokoro_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                           epoch: int, step: int, checkpoint_path: str):
    """
    Saves the current model and optimizer state as a checkpoint.

    Args:
        model (nn.Module): The model instance to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        epoch (int): The current training epoch number.
        step (int): The current global training step number.
        checkpoint_path (str): The file path where the checkpoint will be saved.
    """
    output_dir = osp.dirname(checkpoint_path)
    if output_dir: # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    # Save model, optimizer states, epoch, and step
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }, checkpoint_path)

# --- Default Configuration ---

# This configuration dictionary can be loaded and modified to build the model.
# It reflects the typical parameters for Kokoro-82M.
KOKORO_CONFIG: Dict[str, Union[int, float]] = {
    'n_symbols': 256,         # Number of unique input symbols (e.g., phonemes)
    'num_voices': 54,         # Number of unique speaker identities
    'num_languages': 8,       # Number of supported languages
    'text_dim': 256,          # Dimensionality of text encoder features
    'voice_dim': 256,         # Dimensionality of voice embeddings
    'prosody_dim': 128,       # Dimensionality of predicted prosody features
    'mel_dim': 80,            # Dimensionality of mel-spectrograms (e.g., 80 mel bins)
    'hidden_dim': 512,        # Internal hidden dimension for the decoder
    'num_decoder_layers': 6,  # Number of decoder layers
    'dropout': 0.1            # Dropout rate
}

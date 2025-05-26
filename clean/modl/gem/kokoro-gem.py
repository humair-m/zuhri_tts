import os
import os.path as osp
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

# --- External Utility Imports (Commented out as they are not defined within this scope) ---
# The original context mentioned these, implying they are part of a larger ecosystem
# but are not directly implemented or required for the core Kokoro-82M model definition itself.
# from Utils.ASR.models import ASRCNN             # Likely for Automatic Speech Recognition features
# from Utils.JDC.model import JDCNet              # Likely for Joint Discriminative Coding (e.g., F0 extraction)
# from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator
# These would typically be used for auxiliary tasks or adversarial training in a full TTS system.

# --- Third-party Libraries ---
from munch import Munch # Used for easily accessing dictionary contents as attributes (e.g., config.param)
import yaml            # For loading/saving configuration files (e.g., KOKORO_CONFIG)

"""
================================================================================
Kokoro-82M: A Lightweight and Efficient Text-to-Speech Model Architecture
================================================================================

This module provides a comprehensive definition of the Kokoro-82M Text-to-Speech (TTS) model.
Kokoro-82M is a streamlined adaptation of the StyleTTS-2 architecture, specifically designed
for efficiency and high-quality multi-speaker and multi-lingual speech synthesis.

Key Architectural Innovations and Features:
------------------------------------------
1.  **Lightweight Decoder-Only Design**: Unlike traditional diffusion-based decoders,
    Kokoro-82M employs a simpler, decoder-only architecture. This significantly
    reduces model complexity, computational overhead, and speeds up inference,
    making it suitable for deployment on more moderate hardware.

2.  **Modular Pipeline**: The model is structured as a clear, sequential pipeline
    of specialized neural network modules:
    * `KokoroTextEncoder`: Transforms raw text into rich, contextualized feature representations.
    * `VoicePacketEmbedding`: Manages and integrates distinct speaker identities and language information.
    * `KokoroProsodyPredictor`: Predicts crucial prosodic elements like fundamental frequency (F0),
        energy, and duration, which are vital for natural intonation and expressiveness.
    * `KokoroDecoder`: Generates preliminary mel-spectrograms based on the combined
        text, voice, and prosody features.
    * `KokoroPostnet`: Refines the generated mel-spectrograms, adding fine-grained
        spectral details to enhance perceptual quality.

3.  **Multi-Speaker and Multi-Lingual Capability**: The `VoicePacketEmbedding` module
    explicitly handles multiple voice IDs (around 54 voices) and language IDs (8 languages),
    allowing for flexible speaker switching and multi-lingual synthesis from a single model.

4.  **Efficiency Focus**: The entire design emphasizes computational efficiency,
    enabling the model to be trained effectively on moderate hardware (e.g., NVIDIA A100 80GB GPUs)
    and achieve high performance with a relatively small dataset (less than 100 hours of audio).

5.  **End-to-End Trainable**: The model is optimized end-to-end using supervised learning,
    with loss functions targeting both mel-spectrogram reconstruction and accurate prosody prediction.

This code provides the PyTorch implementation of the Kokoro-82M model, its loss function,
and utility functions for model management (building, loading, saving checkpoints).
"""

class LinearNorm(torch.nn.Module):
    """
    A standard linear (fully connected) layer with a specific weight initialization strategy.
    It applies a linear transformation to the input data: $y = xA^T + b$.

    This class extends `torch.nn.Linear` by applying Xavier uniform initialization
    to the weights, which is a common practice to help in maintaining the scale
    of activations and gradients through the network, especially for layers
    leading to linear or ReLU activations.
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = 'linear'):
        """
        Initializes the LinearNorm layer.

        Args:
            in_dim (int): The number of input features (size of each input sample).
            out_dim (int): The number of output features (size of each output sample).
            bias (bool, optional): If `True`, the layer learns an additive bias. Defaults to `True`.
            w_init_gain (str, optional): The gain value to use for Xavier uniform initialization.
                                         This parameter is passed to `torch.nn.init.calculate_gain`.
                                         'linear' is typically used for layers followed by linear
                                         or no activation, or 'relu' for ReLU activation.
                                         Defaults to 'linear'.
        """
        super(LinearNorm, self).__init__()
        # Instantiate the standard PyTorch linear layer
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # Apply Xavier uniform initialization to the weight tensor.
        # This initializes weights from a uniform distribution, scaled by a gain factor.
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        # Bias is initialized to zeros by default in torch.nn.Linear, which is typically fine.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the LinearNorm layer.

        Args:
            x (torch.Tensor): The input tensor to the linear layer.
                              The last dimension of `x` must be `in_dim`.
                              Shape: `(..., in_dim)`

        Returns:
            torch.Tensor: The output tensor after the linear transformation.
                          The last dimension will be `out_dim`.
                          Shape: `(..., out_dim)`
        """
        return self.linear_layer(x)

class LayerNorm(nn.Module):
    """
    Custom Layer Normalization module.
    This implementation is specifically designed to handle input tensors where
    the channel dimension is *not* the last dimension (e.g., `[B, C, T]` for Conv1d outputs).
    `torch.nn.LayerNorm` typically normalizes over the *last* few dimensions.
    This module transposes the input to bring the `channels` dimension to the last,
    applies `F.layer_norm`, and then transposes it back to the original shape.
    """
    def __init__(self, channels: int, eps: float = 1e-5):
        """
        Initializes the LayerNorm layer.

        Args:
            channels (int): The number of channels in the input tensor (the dimension
                            over which normalization is applied). For an input
                            `[B, C, T]`, `channels` would be `C`.
            eps (float, optional): A small value added to the denominator for numerical stability
                                   during normalization. Defaults to 1e-5.
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        # `gamma` (scale) and `beta` (shift) are learnable parameters that
        # allow the normalized output to be scaled and shifted.
        # They are initialized to 1s and 0s respectively, which means
        # identity transformation initially.
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the LayerNorm layer.

        Args:
            x (torch.Tensor): The input tensor to be normalized.
                              Expected shape: `(batch_size, channels, sequence_length)`
                              or `(batch_size, channels)` for simpler cases.

        Returns:
            torch.Tensor: The normalized output tensor, maintaining the original shape.
                          Shape: `(batch_size, channels, sequence_length)`
        """
        # Transpose the input tensor to move the `channels` dimension to the last position.
        # This is because `F.layer_norm` operates on the last dimension(s) of the input.
        # Example: `[B, C, T]` becomes `[B, T, C]`
        x = x.transpose(1, -1) # `[B, ..., C]`
        
        # Apply layer normalization. `(self.channels,)` specifies the shape of the
        # normalized dimensions. `gamma` and `beta` are applied after normalization.
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        
        # Transpose the tensor back to its original shape.
        # Example: `[B, T, C]` becomes `[B, C, T]`
        return x.transpose(1, -1)

class KokoroTextEncoder(nn.Module):
    """
    The Text Encoder component of the Kokoro-82M model.
    Its primary role is to convert discrete text tokens (e.g., phoneme IDs or character IDs)
    into a continuous sequence of rich, contextualized feature representations.
    This sequence serves as the linguistic input for subsequent stages of speech synthesis.

    The encoder employs a hybrid architecture:
    1.  **Embedding Layer**: Maps each discrete symbol ID to a dense vector.
    2.  **1D Convolutional Layers**: A stack of CNNs to capture local patterns and
        n-gram information within the text sequence. `weight_norm` is used for stability.
    3.  **Bidirectional LSTM**: A single LSTM layer processes the CNN outputs to
        model long-range dependencies and provide a global context for each text unit.
    """
    def __init__(self, channels: int = 256, kernel_size: int = 5, depth: int = 4, n_symbols: int = 256, dropout: float = 0.1):
        """
        Initializes the KokoroTextEncoder.

        Args:
            channels (int, optional): The dimensionality of the feature representations
                                      that flow through the encoder. This is the output
                                      dimension of the embedding, CNNs, and LSTM. Defaults to 256.
            kernel_size (int, optional): The size of the convolutional filters in the CNN layers.
                                         A common choice for capturing local patterns. Defaults to 5.
            depth (int, optional): The number of sequential 1D convolutional blocks.
                                   A "reduced depth" (default 4) contributes to the
                                   "lightweight" nature of Kokoro-82M. Defaults to 4.
            n_symbols (int, optional): The total number of unique discrete input symbols
                                       (e.g., phonemes, characters, punctuation) that the
                                       embedding layer needs to map. Defaults to 256.
            dropout (float, optional): The dropout probability applied after activation functions
                                       within the CNN blocks, and for the LSTM output.
                                       Helps prevent overfitting. Defaults to 0.1.
        """
        super().__init__()
        self.channels = channels
        self.n_symbols = n_symbols

        # 1. Embedding Layer: Converts integer symbol IDs into dense, continuous vectors.
        # `n_symbols` is the size of the dictionary of embeddings.
        # `channels` is the size of each embedding vector.
        self.embedding = nn.Embedding(n_symbols, channels)
        
        # 2. 1D Convolutional Layers: A list of `nn.Sequential` blocks.
        # `padding` ensures that the output sequence length remains the same as the input
        # after convolution, which is important for alignment with the LSTM.
        padding = (kernel_size - 1) // 2 
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                # `weight_norm` is applied to the convolutional layer. It reparameterizes
                # the weights to decouple their magnitude from their direction, which can
                # lead to faster convergence and better performance.
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels), # Custom LayerNorm for channel-first input (B, C, T)
                nn.ReLU(inplace=True), # Rectified Linear Unit activation; `inplace=True` saves memory
                nn.Dropout(dropout),   # Dropout for regularization
            ))

        # 3. Bidirectional LSTM Layer: Processes the sequence of features from the CNNs.
        # `channels` is the input feature size for the LSTM.
        # `channels // 2` is the hidden state size for *each* direction. Since it's bidirectional,
        # the concatenated output will be `(channels // 2) * 2 = channels`.
        # `num_layers=1` indicates a single LSTM layer, contributing to the "lightweight" design.
        # `batch_first=True` means input/output tensors are `(batch, sequence, features)`.
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)
        
        # Output Projection: A linear layer to map the LSTM's output back to `channels`.
        # While the bidirectional LSTM already outputs `channels`, this layer can
        # serve as an additional non-linearity or transformation if needed.
        self.output_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass of the KokoroTextEncoder.

        Args:
            x (torch.Tensor): Input batch of text token IDs.
                              Shape: `(batch_size, text_sequence_length)`
            input_lengths (torch.Tensor, optional): A 1D tensor specifying the actual
                                                   (unpadded) length of each sequence in the batch.
                                                   Crucial for `pack_padded_sequence` with LSTM.
                                                   Shape: `(batch_size,)`
                                                   Defaults to `None`.
            mask (torch.Tensor, optional): A boolean tensor, where `True` indicates padded positions
                                           that should be ignored (e.g., set to zero).
                                           Shape: `(batch_size, text_sequence_length)`
                                           Defaults to `None`.

        Returns:
            torch.Tensor: The encoded text features, ready for the next stages of the TTS pipeline.
                          Shape: `(batch_size, text_sequence_length, channels)`
        """
        # 1. Embedding Lookup: Convert integer IDs to dense vectors.
        x = self.embedding(x)  # Shape: `[B, T_text, channels]` (Batch, Text Length, Feature Dimension)
        
        # 2. Transpose for CNNs: `nn.Conv1d` expects input in `(batch, channels, sequence_length)` format.
        x = x.transpose(1, 2)  # Shape: `[B, channels, T_text]`

        # 3. Apply Mask (if provided) before CNNs: Zero out features corresponding to padding.
        if mask is not None:
            # Expand mask to match CNN input dimensions for broadcasting.
            # `unsqueeze(1)` adds a channel dimension to the mask.
            mask_expanded = mask.to(x.device).unsqueeze(1) # Shape: `[B, 1, T_text]`
            x.masked_fill_(mask_expanded, 0.0) # Fill masked positions with zeros
        
        # 4. Pass through CNN layers: Iterate over the `ModuleList` of CNN blocks.
        for c in self.cnn:
            x = c(x)
            # Re-apply mask after each CNN layer to ensure padded regions remain zero
            # (e.g., if ReLU or other operations introduce non-zeros in padded areas).
            if mask is not None:
                x.masked_fill_(mask_expanded, 0.0)
                
        # 5. Transpose back for LSTM: `nn.LSTM` with `batch_first=True` expects
        # input in `(batch, sequence_length, features)` format.
        x = x.transpose(1, 2)  # Shape: `[B, T_text, channels]`

        # 6. Pack Padded Sequence (if lengths provided):
        # `nn.utils.rnn.pack_padded_sequence` is crucial for efficient LSTM processing
        # by avoiding computation on padded elements.
        # IMPORTANT: `input_lengths` must be on CPU for this function. This introduces
        # a CPU-GPU synchronization point, which can be a bottleneck.
        if input_lengths is not None:
            input_lengths_cpu = input_lengths.cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths_cpu, batch_first=True, enforce_sorted=False)

        # 7. Pass through LSTM:
        # `flatten_parameters()` is a utility for `nn.LSTM` that makes parameters
        # contiguous in memory, which can improve performance, especially with DataParallel.
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # `_` captures hidden/cell states, which are not used here.
        
        # 8. Pad Packed Sequence Back (if packed):
        # If the sequence was packed, `nn.utils.rnn.pad_packed_sequence` restores it
        # to a padded tensor, making it compatible with subsequent modules.
        if input_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        # 9. Output Projection: Apply the final linear transformation.
        x = self.output_proj(x) # Shape: `[B, T_text, channels]`
        return x

class VoicePacketEmbedding(nn.Module):
    """
    This module is responsible for creating a unified "voice packet" embedding.
    This embedding captures both the speaker's identity and the language spoken,
    allowing the TTS model to generate speech conditioned on specific voices
    and languages. It supports multi-speaker and multi-lingual synthesis.
    """
    def __init__(self, num_voices: int = 54, voice_dim: int = 256):
        """
        Initializes the VoicePacketEmbedding module.

        Args:
            num_voices (int, optional): The total number of distinct voice (speaker) identities
                                        that the model is trained to synthesize. Defaults to 54.
            voice_dim (int, optional): The target dimensionality of the final combined
                                       voice packet embedding. Defaults to 256.
        """
        super().__init__()
        self.num_voices = num_voices
        self.voice_dim = voice_dim
        
        # Embedding layer for individual voice IDs.
        # Each voice ID (integer) is mapped to a dense vector of `voice_dim` size.
        self.voice_embeddings = nn.Embedding(num_voices, voice_dim)
        
        # Embedding layer for language IDs.
        # The description mentions 8 languages, so the number of embeddings is 8.
        # The dimensionality of the language embedding is a design choice, here `voice_dim // 4`.
        self.language_embeddings = nn.Embedding(8, voice_dim // 4) # Assuming 8 languages
        
        # A linear projection layer to combine the voice and language embeddings.
        # The input dimension is the sum of `voice_dim` and `voice_dim // 4`.
        # The output dimension is `voice_dim`.
        self.proj = nn.Linear(voice_dim + voice_dim // 4, voice_dim)
        
    def forward(self, voice_id: torch.Tensor, language_id: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass to generate the voice packet embedding.

        Args:
            voice_id (torch.Tensor): A tensor containing the voice ID(s).
                                     Can be a single scalar integer (for inference)
                                     or a batch of integers.
                                     Shape: `(batch_size,)` or `()` (scalar).
            language_id (torch.Tensor, optional): A tensor containing the language ID(s).
                                                  Can be a single scalar integer or a batch.
                                                  If `None`, only the voice embedding is used.
                                                  Shape: `(batch_size,)` or `()` (scalar).
                                                  Defaults to `None`.

        Returns:
            torch.Tensor: The combined voice and (optionally) language embedding.
                          Shape: `(batch_size, voice_dim)` if batch input,
                          or `(voice_dim,)` if scalar input.
        """
        # Retrieve the voice embedding(s) based on the provided voice_id(s).
        voice_emb = self.voice_embeddings(voice_id) # Shape: `[B, voice_dim]` or `[voice_dim]`
        
        if language_id is not None:
            # Retrieve the language embedding(s).
            lang_emb = self.language_embeddings(language_id) # Shape: `[B, voice_dim // 4]` or `[voice_dim // 4]`

            # Ensure `lang_emb` has a batch dimension if the input `language_id` was a scalar.
            # This is crucial for consistent concatenation with `voice_emb` when `voice_emb`
            # might have a batch dimension (e.g., during training with batches) while
            # `language_id` is provided as a single scalar.
            if lang_emb.dim() == 0: # If language_id was a scalar, embedding output is `[D_lang]` (0-dim tensor)
                lang_emb = lang_emb.unsqueeze(0) # Make it `[1, D_lang]`
            elif lang_emb.dim() == 1 and voice_emb.dim() == 2: # If lang_emb is [D_lang] and voice_emb is [B, D_voice]
                 lang_emb = lang_emb.unsqueeze(0) # Make it [1, D_lang] to match batch dimension for concat
            
            # Concatenate the voice and language embeddings along the last dimension (feature dimension).
            # Example: `[B, voice_dim]` + `[B, voice_dim//4]` -> `[B, voice_dim + voice_dim//4]`
            combined = torch.cat([voice_emb, lang_emb], dim=-1)
            
            # Project the combined embedding to the final desired `voice_dim`.
            return self.proj(combined)
            
        # If no language ID is provided, return only the voice embedding.
        return voice_emb

class KokoroProsodyPredictor(nn.Module):
    """
    The Prosody Predictor component of Kokoro-82M.
    This module is responsible for predicting crucial prosodic features:
    Fundamental Frequency (F0), Energy, and Duration. These features are
    essential for synthesizing speech with natural intonation, rhythm, and expressiveness.

    It takes the encoded text features and the voice packet embedding as input,
    and uses a feed-forward network with separate heads for each prosodic feature.
    """
    def __init__(self, text_dim: int = 256, voice_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        """
        Initializes the KokoroProsodyPredictor.

        Args:
            text_dim (int, optional): The dimensionality of the input text features
                                      from the `KokoroTextEncoder`. Defaults to 256.
            voice_dim (int, optional): The dimensionality of the input voice packet embedding
                                       from the `VoicePacketEmbedding`. Defaults to 256.
            hidden_dim (int, optional): The internal hidden dimensionality used within the
                                        prosody prediction network. This is the output dimension
                                        for the initial `text_proj` and `voice_proj` layers.
                                        Defaults to 256.
            dropout (float, optional): The dropout probability applied within the network.
                                       Defaults to 0.1.
        """
        super().__init__()
        
        # Linear projections to transform input `text_features` and `voice_embedding`
        # into a common `hidden_dim` before they are concatenated.
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.voice_proj = nn.Linear(voice_dim, hidden_dim)
        
        # The main prosody prediction network. It's a simple multi-layer perceptron (MLP).
        # The input dimension to the first linear layer is `hidden_dim * 2` because
        # it concatenates the projected text and voice features.
        self.prosody_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # First layer: expands to hidden_dim
            nn.ReLU(inplace=True),               # ReLU activation
            nn.Dropout(dropout),                 # Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim // 2), # Second layer: reduces dimension
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Separate linear heads for predicting each specific prosodic feature.
        # Each head takes the output of `prosody_net` (`hidden_dim // 2`) and
        # projects it to a single scalar value per time step.
        self.f0_head = nn.Linear(hidden_dim // 2, 1)  # Predicts F0 (Fundamental Frequency)
        self.energy_head = nn.Linear(hidden_dim // 2, 1)  # Predicts Energy (loudness)
        self.duration_head = nn.Linear(hidden_dim // 2, 1)  # Predicts Duration
        
    def forward(self, text_features: torch.Tensor, voice_embedding: torch.Tensor) -> dict:
        """
        Performs the forward pass of the Prosody Predictor.

        Args:
            text_features (torch.Tensor): Encoded text features from `KokoroTextEncoder`.
                                          Shape: `(batch_size, text_sequence_length, text_dim)`
            voice_embedding (torch.Tensor): Combined voice/language embedding from `VoicePacketEmbedding`.
                                            Shape: `(batch_size, voice_dim)` (or `(voice_dim,)` for scalar input).

        Returns:
            dict: A dictionary containing the predicted prosodic features and an
                  intermediate feature tensor used by the decoder:
                  - 'f0' (torch.Tensor): Predicted F0 values for each text unit.
                                         Shape: `(batch_size, text_sequence_length)`
                  - 'energy' (torch.Tensor): Predicted energy values for each text unit.
                                             Shape: `(batch_size, text_sequence_length)`
                  - 'duration' (torch.Tensor): Predicted duration values for each text unit.
                                               `sigmoid` activated to be between 0 and 1.
                                               Shape: `(batch_size, text_sequence_length)`
                  - 'prosody_features' (torch.Tensor): Intermediate features from `prosody_net`.
                                                       These are passed to the `KokoroDecoder`.
                                                       Shape: `(batch_size, text_sequence_length, hidden_dim // 2)`
        """
        # Project input text features and voice embedding to the common `hidden_dim`.
        text_proj = self.text_proj(text_features)  # Shape: `[B, T_text, hidden_dim]`
        voice_proj = self.voice_proj(voice_embedding)  # Shape: `[B, hidden_dim]` or `[hidden_dim]`
        
        # Expand `voice_proj` to match the sequence length of `text_proj`.
        # This is done by unsqueezing a dimension and then expanding it.
        # This is an efficient broadcasting operation, avoiding large memory copies.
        if voice_proj.dim() == 0: # Handle scalar voice_embedding (e.g., single sample inference)
            voice_proj = voice_proj.unsqueeze(0) # -> `[1]`
        if voice_proj.dim() == 1: # If voice_proj is `[D]`, make it `[1, D]`
            voice_proj = voice_proj.unsqueeze(0)
        voice_proj = voice_proj.unsqueeze(1).expand(-1, text_proj.size(1), -1) # -> `[B, T_text, hidden_dim]`
            
        # Concatenate the projected text and voice features along the feature dimension.
        combined = torch.cat([text_proj, voice_proj], dim=-1)  # Shape: `[B, T_text, hidden_dim * 2]`
        
        # Pass the combined features through the prosody prediction network.
        prosody_features = self.prosody_net(combined) # Shape: `[B, T_text, hidden_dim // 2]`
        
        # Predict F0, energy, and duration using their respective heads.
        # `.squeeze(-1)` removes the last dimension if it's 1 (e.g., `[B, T, 1]` -> `[B, T]`).
        f0 = self.f0_head(prosody_features).squeeze(-1)  # Shape: `[B, T_text]`
        energy = self.energy_head(prosody_features).squeeze(-1)  # Shape: `[B, T_text]`
        # Apply sigmoid activation to duration predictions. This constrains the output
        # to a range (0, 1), which can be useful if duration is represented as a ratio
        # or needs to be positive.
        duration = torch.sigmoid(self.duration_head(prosody_features)).squeeze(-1)  # Shape: `[B, T_text]`
        
        return {
            'f0': f0,
            'energy': energy,
            'duration': duration,
            'prosody_features': prosody_features # This tensor is crucial for the `KokoroDecoder`
        }

class KokoroDecoder(nn.Module):
    """
    The lightweight decoder-only model for mel-spectrogram generation in Kokoro-82M.
    This module synthesizes mel-spectrograms, which are spectral representations of audio,
    from the combined linguistic (text), speaker/language (voice), and prosodic features.

    It employs a stack of simplified transformer-like decoder layers, allowing it to
    effectively integrate diverse input information and generate high-quality spectrograms.
    """
    def __init__(self, 
                 text_dim: int = 256, 
                 voice_dim: int = 256, 
                 prosody_dim: int = 128, # This is the actual dimension of 'prosody_features' input
                 mel_dim: int = 80, 
                 hidden_dim: int = 512,
                 num_layers: int = 6,
                 dropout: float = 0.1):
        """
        Initializes the KokoroDecoder.

        Args:
            text_dim (int, optional): Dimensionality of the input text features. Defaults to 256.
            voice_dim (int, optional): Dimensionality of the input voice packet embeddings. Defaults to 256.
            prosody_dim (int, optional): Dimensionality of the input prosody features (from `ProsodyPredictor`).
                                         This should match the `hidden_dim // 2` output of `prosody_net`.
                                         Defaults to 128.
            mel_dim (int, optional): The dimensionality of the output mel-spectrogram features
                                     (e.g., number of mel-bins). Defaults to 80.
            hidden_dim (int, optional): The internal hidden dimensionality used within each
                                        `KokoroDecoderLayer`. This is also the dimension
                                        after the initial `input_proj`. Defaults to 512.
            num_layers (int, optional): The number of stacked `KokoroDecoderLayer`s.
                                        A "reduced" number (default 6) contributes to
                                        the lightweight nature. Defaults to 6.
            dropout (float, optional): Dropout probability applied within decoder layers. Defaults to 0.1.
        """
        super().__init__()
        
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        
        # Input Projection Layer: Combines the three main input feature streams
        # (text, voice, prosody) into a single tensor with the decoder's `hidden_dim`.
        # The input size is the sum of the dimensions of these three feature types.
        self.input_proj = nn.Linear(text_dim + voice_dim + prosody_dim, hidden_dim)
        
        # Stack of Decoder Layers: These are the core computational blocks of the decoder.
        # Each layer processes the sequence, applying self-attention and feed-forward networks.
        self.decoder_layers = nn.ModuleList([
            KokoroDecoderLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Output Projection Layer: Transforms the final hidden representations from the
        # decoder layers into the mel-spectrogram format.
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), # Reduce dimension
            nn.ReLU(inplace=True),                 # ReLU activation
            nn.Dropout(dropout),                   # Dropout
            nn.Linear(hidden_dim // 2, mel_dim)    # Final projection to `mel_dim`
        )
        
        # Postnet: A separate network for refining the generated mel-spectrogram.
        # It adds a residual to the initial mel output for improved quality.
        self.postnet = KokoroPostnet(mel_dim, hidden_dim // 4) # `hidden_dim // 4` is a design choice
        
    def forward(self, 
                text_features: torch.Tensor, 
                voice_embedding: torch.Tensor, 
                prosody_features: torch.Tensor, 
                lengths: torch.Tensor = None) -> dict:
        """
        Performs the forward pass of the KokoroDecoder.

        Args:
            text_features (torch.Tensor): Encoded text features from `KokoroTextEncoder`.
                                          Shape: `(batch_size, text_sequence_length, text_dim)`
            voice_embedding (torch.Tensor): Voice/language embedding from `VoicePacketEmbedding`.
                                            Shape: `(batch_size, voice_dim)`
            prosody_features (torch.Tensor): Intermediate prosody features from `KokoroProsodyPredictor`.
                                             Shape: `(batch_size, text_sequence_length, prosody_dim)`
            lengths (torch.Tensor, optional): A 1D tensor specifying the actual (unpadded) length
                                              of each sequence in the batch. Used for attention masking
                                              within `KokoroDecoderLayer`. Shape: `(batch_size,)`.
                                              Defaults to `None`.

        Returns:
            dict: A dictionary containing the generated mel-spectrograms:
                  - 'mel_before' (torch.Tensor): The mel-spectrogram generated directly
                                                 by the decoder before Postnet refinement.
                                                 Shape: `(batch_size, mel_sequence_length, mel_dim)`
                  - 'mel_after' (torch.Tensor): The final, refined mel-spectrogram after
                                                the Postnet has applied its residual.
                                                Shape: `(batch_size, mel_sequence_length, mel_dim)`
        """
        # Get batch size and sequence length from text features for consistent expansion.
        batch_size, seq_len, _ = text_features.shape
        
        # Expand `voice_embedding` to match the sequence length of text and prosody features.
        # This broadcasting ensures the voice information is available at every time step.
        if voice_embedding.dim() == 0: # Handle scalar voice_embedding
            voice_embedding = voice_embedding.unsqueeze(0)
        if voice_embedding.dim() == 1: # If voice_embedding is `[D]`, make it `[1, D]`
            voice_embedding = voice_embedding.unsqueeze(0)
        voice_embedding = voice_embedding.unsqueeze(1).expand(-1, seq_len, -1) # -> `[B, T_text, voice_dim]`
            
        # Concatenate all three input feature streams along the feature dimension.
        # This combined tensor serves as the primary input to the decoder's layers.
        combined_features = torch.cat([text_features, voice_embedding, prosody_features], dim=-1)
        
        # Project the combined features to the decoder's internal `hidden_dim`.
        x = self.input_proj(combined_features)  # Shape: `[B, T_text, hidden_dim]`
        
        # Pass the processed features through each stacked decoder layer.
        for layer in self.decoder_layers:
            x = layer(x, lengths) # `lengths` are passed to enable attention masking within layers
            
        # Generate the initial mel-spectrogram from the decoder's final hidden representations.
        mel_before = self.output_proj(x)  # Shape: `[B, T_mel, mel_dim]`
        
        # Apply the Postnet for refinement. The Postnet predicts a residual, which is
        # added to `mel_before` to produce the final, higher-quality mel-spectrogram.
        mel_after = self.postnet(mel_before) + mel_before
        
        return {
            'mel_before': mel_before,
            'mel_after': mel_after
        }

class KokoroDecoderLayer(nn.Module):
    """
    A single building block of the `KokoroDecoder`.
    This layer is inspired by a transformer decoder block, featuring:
    1.  **Multi-head Self-Attention**: Allows the model to weigh the importance of
        different parts of the input sequence to itself.
    2.  **Position-wise Feed-Forward Network (FFN)**: A simple MLP applied independently
        to each position in the sequence, enhancing feature transformations.
    Both sub-layers incorporate residual connections and layer normalization for stable
    training and improved performance.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1, num_heads: int = 8):
        """
        Initializes a single KokoroDecoderLayer.

        Args:
            hidden_dim (int): The feature dimension that flows through this layer.
                              This is the input and output dimension of the layer.
            dropout (float, optional): The dropout probability applied after attention
                                       and within the FFN. Defaults to 0.1.
            num_heads (int, optional): The number of attention heads in the `MultiheadAttention` module.
                                       More heads allow the model to focus on different aspects
                                       of the input simultaneously. Defaults to 8.
        """
        super().__init__()
        
        # Multi-head self-attention module.
        # `batch_first=True` means input/output tensors are `(batch, sequence, feature)`.
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, # The feature dimension
            num_heads=num_heads,  # Number of parallel attention heads
            dropout=dropout,      # Dropout applied to attention weights
            batch_first=True      # Input/output tensors are batch-first
        )
        # Layer Normalization applied after the self-attention block.
        self.norm1 = nn.LayerNorm(hidden_dim) 
        
        # Position-wise Feed-Forward Network (FFN).
        # This is a simple two-layer MLP applied independently to each position.
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), # Expand dimension (e.g., to 2x `hidden_dim`)
            nn.ReLU(inplace=True),                 # ReLU activation
            nn.Dropout(dropout),                   # Dropout
            nn.Linear(hidden_dim * 2, hidden_dim), # Project back to original `hidden_dim`
            nn.Dropout(dropout)                    # Dropout
        )
        # Layer Normalization applied after the FFN block.
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Separate dropout layer for the residual connections.
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass of a single KokoroDecoderLayer.

        Args:
            x (torch.Tensor): The input tensor to this decoder layer.
                              Shape: `(batch_size, sequence_length, hidden_dim)`
            lengths (torch.Tensor, optional): A 1D tensor specifying the actual (unpadded) length
                                              of each sequence in the batch. Used to create a padding
                                              mask for the self-attention mechanism, preventing attention
                                              to padding tokens. Shape: `(batch_size,)`.
                                              Defaults to `None`.

        Returns:
            torch.Tensor: The output tensor from this decoder layer, maintaining the same shape as input.
                          Shape: `(batch_size, sequence_length, hidden_dim)`
        """
        # 1. Create Attention Mask (if sequence lengths are provided):
        # This mask prevents the attention mechanism from attending to padded positions,
        # ensuring that padding does not influence the output.
        attn_mask = None
        if lengths is not None:
            max_len = x.size(1) # Get the maximum sequence length in the current batch
            # `key_padding_mask` for `MultiheadAttention` expects a boolean tensor of shape `(N, S_key)`.
            # `True` indicates positions that should be masked (ignored).
            # `torch.arange(max_len).unsqueeze(0)` creates `[[0, 1, ..., max_len-1]]`.
            # `lengths.unsqueeze(1)` creates `[[L1], [L2], ...]`.
            # The comparison `... >= ...` results in `True` for indices greater than or equal to the length,
            # effectively marking padded positions.
            attn_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            
        # 2. Self-Attention Block:
        # Applies multi-head self-attention. `x` serves as query, key, and value.
        # `key_padding_mask` is used to mask out padded elements in the key/value sequences.
        residual = x # Store input for residual connection
        x_attn, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        # Add residual connection, apply dropout, and then Layer Normalization.
        x = self.norm1(residual + self.dropout_layer(x_attn))
        
        # 3. Feed-Forward Network (FFN) Block:
        # Applies the position-wise FFN.
        residual = x # Store input for residual connection
        x_ffn = self.ffn(x)
        # Add residual connection and then Layer Normalization.
        x = self.norm2(residual + x_ffn)
        
        return x

class KokoroPostnet(nn.Module):
    """
    The Postnet module for mel-spectrogram refinement.
    This component is applied after the main `KokoroDecoder` to enhance the quality
    and detail of the generated mel-spectrograms. It acts as a residual network,
    predicting a correction term that is added to the initial mel-spectrogram output.

    It consists of a stack of 1D convolutional layers, typically with BatchNorm,
    Tanh activations (except for the last layer), and Dropout.
    """
    def __init__(self, mel_dim: int = 80, hidden_dim: int = 128, num_layers: int = 5, kernel_size: int = 5):
        """
        Initializes the KokoroPostnet.

        Args:
            mel_dim (int, optional): The dimensionality of the input and output mel-spectrogram features.
                                     Defaults to 80 (e.g., 80 mel-bins).
            hidden_dim (int, optional): The internal hidden dimensionality used within the
                                        convolutional layers of the Postnet. Defaults to 128.
            num_layers (int, optional): The total number of 1D convolutional layers in the Postnet.
                                        Defaults to 5.
            kernel_size (int, optional): The size of the convolutional filters. Defaults to 5.
        """
        super().__init__()
        
        # Calculate padding to ensure that the output sequence length remains the same
        # as the input sequence length after convolution.
        padding = (kernel_size - 1) // 2
        
        self.convs = nn.ModuleList()
        
        # First convolutional layer: Transforms `mel_dim` input to `hidden_dim`.
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(mel_dim, hidden_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_dim), # Batch Normalization for stable training
                nn.Tanh(),                  # Tanh activation, common in Postnets
                nn.Dropout(0.5)             # Dropout for regularization
            )
        )
        
        # Intermediate convolutional layers: Maintain `hidden_dim` throughout.
        # `num_layers - 2` because the first and last layers are defined separately.
        for _ in range(num_layers - 2):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5)
                )
            )
            
        # Last convolutional layer: Transforms `hidden_dim` back to `mel_dim`.
        # No Tanh activation here, as this layer predicts a residual value that
        # will be added to the original mel-spectrogram.
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(hidden_dim, mel_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(0.5)
            )
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the Postnet.

        Args:
            x (torch.Tensor): The mel-spectrogram generated by the `KokoroDecoder`
                              (before Postnet refinement).
                              Shape: `(batch_size, mel_sequence_length, mel_dim)`

        Returns:
            torch.Tensor: The predicted residual for mel-spectrogram refinement.
                          This tensor has the same shape as the input `x` and is
                          intended to be added to `x` to produce the final mel-spectrogram.
                          Shape: `(batch_size, mel_sequence_length, mel_dim)`
        """
        # Transpose the input tensor from `(B, T_mel, mel_dim)` to `(B, mel_dim, T_mel)`.
        # This is necessary because `nn.Conv1d` expects the channel dimension to be second.
        x = x.transpose(1, 2)
        
        # Pass the tensor through each sequential convolutional block.
        for conv in self.convs:
            x = conv(x)
            
        # Transpose the tensor back to its original `(B, T_mel, mel_dim)` shape
        # for consistency with the rest of the model and subsequent operations.
        x = x.transpose(1, 2)
        return x

class KokoroTTS(nn.Module):
    """
    The complete Kokoro-82M Text-to-Speech (TTS) model.
    This class orchestrates all the individual sub-modules to perform
    end-to-end speech synthesis, converting input text into mel-spectrograms.
    The mel-spectrograms can then be converted to audible waveforms by a separate vocoder.

    The model supports multi-speaker and multi-lingual synthesis by incorporating
    voice and language embeddings. It also predicts prosodic features to ensure
    natural and expressive speech.
    """
    def __init__(self, 
                 n_symbols: int = 256,
                 num_voices: int = 54,
                 num_languages: int = 8,
                 text_dim: int = 256,
                 voice_dim: int = 256,
                 prosody_dim: int = 128, # Output dimension of prosody features from predictor
                 mel_dim: int = 80,
                 hidden_dim: int = 512,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1):
        """
        Initializes the KokoroTTS model with specified hyperparameters.

        Args:
            n_symbols (int, optional): The total number of unique input text symbols (e.g., phonemes). Defaults to 256.
            num_voices (int, optional): The total number of distinct speaker identities the model can synthesize. Defaults to 54.
            num_languages (int, optional): The total number of languages supported. This is used internally by `VoicePacketEmbedding`. Defaults to 8.
            text_dim (int, optional): The feature dimensionality for the `KokoroTextEncoder`'s output. Defaults to 256.
            voice_dim (int, optional): The feature dimensionality for the `VoicePacketEmbedding`'s output. Defaults to 256.
            prosody_dim (int, optional): The desired output feature dimensionality for the `prosody_features`
                                         from the `KokoroProsodyPredictor`. Defaults to 128.
            mel_dim (int, optional): The dimensionality of the output mel-spectrograms (e.g., number of mel-bins). Defaults to 80.
            hidden_dim (int, optional): The internal hidden dimensionality used within the `KokoroDecoder` layers. Defaults to 512.
            num_decoder_layers (int, optional): The number of stacked `KokoroDecoderLayer`s. Defaults to 6.
            dropout (float, optional): The general dropout probability applied across various modules for regularization. Defaults to 0.1.
        """
        super().__init__()
        
        # Initialize the core components of the TTS pipeline:
        
        # 1. Text Encoder: Converts input text tokens into a sequence of linguistic features.
        self.text_encoder = KokoroTextEncoder(
            channels=text_dim, 
            n_symbols=n_symbols,
            dropout=dropout
        )
        
        # 2. Voice Packet Embedding: Generates combined speaker and language embeddings.
        self.voice_embedding = VoicePacketEmbedding(
            num_voices=num_voices,
            voice_dim=voice_dim
        )
        
        # 3. Prosody Predictor: Predicts F0, energy, and duration, and provides
        #    intermediate prosody features for the decoder.
        #    Note: The `hidden_dim` for the prosody predictor's internal network is set
        #    to `prosody_dim * 2` so that its final `prosody_features` output has `prosody_dim`.
        self.prosody_predictor = KokoroProsodyPredictor(
            text_dim=text_dim,
            voice_dim=voice_dim,
            hidden_dim=prosody_dim * 2, 
            dropout=dropout
        )
        
        # 4. Decoder: Generates mel-spectrograms from text, voice, and prosody features.
        self.decoder = KokoroDecoder(
            text_dim=text_dim,
            voice_dim=voice_dim,
            prosody_dim=prosody_dim, # This must match the actual dimension of prosody_predictor's output features
            mel_dim=mel_dim,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
    def forward(self, 
                text_tokens: torch.Tensor, 
                voice_ids: torch.Tensor, 
                language_ids: torch.Tensor = None, 
                text_lengths: torch.Tensor = None) -> dict:
        """
        Defines the forward pass for training the Kokoro-82M model.
        This method processes a batch of inputs and returns all relevant predictions.

        Args:
            text_tokens (torch.Tensor): Batch of text token ID sequences.
                                        Shape: `(batch_size, max_text_sequence_length)`
            voice_ids (torch.Tensor): Batch of voice IDs.
                                      Shape: `(batch_size,)`
            language_ids (torch.Tensor, optional): Batch of language IDs.
                                                   Shape: `(batch_size,)`. Defaults to `None`.
            text_lengths (torch.Tensor, optional): A 1D tensor specifying the actual (unpadded)
                                                   length of each text sequence in the batch.
                                                   Used for padding in `TextEncoder` and attention
                                                   masking in `Decoder`. Shape: `(batch_size,)`.
                                                   Defaults to `None`.

        Returns:
            dict: A dictionary containing all model predictions:
                  - 'mel_before' (torch.Tensor): Mel-spectrograms generated before Postnet.
                                                 Shape: `(B, T_mel, mel_dim)`
                  - 'mel_after' (torch.Tensor): Final mel-spectrograms after Postnet refinement.
                                                Shape: `(B, T_mel, mel_dim)`
                  - 'f0' (torch.Tensor): Predicted F0 values for each text unit.
                                         Shape: `(B, T_text)`
                  - 'energy' (torch.Tensor): Predicted energy values for each text unit.
                                             Shape: `(B, T_text)`
                  - 'duration' (torch.Tensor): Predicted duration values for each text unit.
                                               Shape: `(B, T_text)`
        """
        # 1. Encode text tokens: Convert text into a sequence of rich linguistic features.
        text_features = self.text_encoder(text_tokens, text_lengths) # Output shape: `[B, T_text, text_dim]`
        
        # 2. Get voice embeddings: Generate combined speaker and language conditioning.
        voice_embeddings = self.voice_embedding(voice_ids, language_ids) # Output shape: `[B, voice_dim]`
        
        # 3. Predict prosody: Forecast F0, energy, and duration, and get intermediate prosody features.
        prosody_output = self.prosody_predictor(text_features, voice_embeddings)
        # `prosody_output` contains 'f0', 'energy', 'duration', and 'prosody_features'
        
        # 4. Generate mel-spectrogram: Use text, voice, and prosody features to synthesize mel-spectrograms.
        decoder_output = self.decoder(
            text_features, 
            voice_embeddings, 
            prosody_output['prosody_features'], # Pass the intermediate prosody features to the decoder
            text_lengths # Pass text_lengths to the decoder for attention masking
        )
        # `decoder_output` contains 'mel_before' and 'mel_after'
        
        # Return all relevant predictions as a dictionary
        return {
            'mel_before': decoder_output['mel_before'],
            'mel_after': decoder_output['mel_after'],
            'f0': prosody_output['f0'],
            'energy': prosody_output['energy'],
            'duration': prosody_output['duration']
        }
    
    def inference(self, 
                  text_tokens: torch.Tensor, 
                  voice_id: torch.Union[int, torch.Tensor], 
                  language_id: torch.Union[int, torch.Tensor] = None) -> dict:
        """
        Performs inference (speech synthesis) for a single input sample.
        This method is designed for generating speech from a single text input,
        a single speaker ID, and an optional single language ID.
        It automatically sets the model to evaluation mode (`self.eval()`) and disables
        gradient computation (`torch.no_grad()`) for efficient inference.

        Args:
            text_tokens (torch.Tensor): The input text token sequence.
                                        Can be a 1D tensor `(text_sequence_length,)`
                                        or a 2D tensor `(1, text_sequence_length)`.
            voice_id (Union[int, torch.Tensor]): The ID of the desired speaker.
                                                 Can be a Python `int` or a 0-dimensional `torch.Tensor`.
            language_id (Union[int, torch.Tensor], optional): The ID of the desired language.
                                                               Can be a Python `int` or a 0-dimensional `torch.Tensor`.
                                                               Defaults to `None`.

        Returns:
            dict: A dictionary containing the predicted mel-spectrograms and prosodic features
                  for the single synthesized sample. All output tensors will have a batch dimension of 1.
                  - 'mel_before' (torch.Tensor): Shape: `(1, T_mel, mel_dim)`
                  - 'mel_after' (torch.Tensor): Shape: `(1, T_mel, mel_dim)`
                  - 'f0' (torch.Tensor): Shape: `(1, T_text)`
                  - 'energy' (torch.Tensor): Shape: `(1, T_text)`
                  - 'duration' (torch.Tensor): Shape: `(1, T_text)`
        """
        # Set the model to evaluation mode. This disables dropout layers and
        # sets batch normalization layers to use their running means and variances.
        self.eval() 
        
        # Disable gradient computation. This reduces memory consumption and speeds up computation,
        # as gradients are not needed during inference.
        with torch.no_grad(): 
            # Ensure `voice_id` is a 1-element tensor on the correct device.
            if isinstance(voice_id, int):
                voice_id_tensor = torch.tensor([voice_id], device=text_tokens.device)
            elif voice_id.dim() == 0: # If it's a 0-dim tensor, unsqueeze to add batch dim
                voice_id_tensor = voice_id.unsqueeze(0)
            else: # Already a 1D tensor, assume it's `[ID]`
                voice_id_tensor = voice_id

            # Ensure `language_id` is a 1-element tensor on the correct device, if provided.
            language_id_tensor = None
            if language_id is not None:
                if isinstance(language_id, int):
                    language_id_tensor = torch.tensor([language_id], device=text_tokens.device)
                elif language_id.dim() == 0: # If it's a 0-dim tensor, unsqueeze to add batch dim
                    language_id_tensor = language_id.unsqueeze(0)
                else: # Already a 1D tensor, assume it's `[ID]`
                    language_id_tensor = language_id
                
            # Ensure `text_tokens` has a batch dimension of 1 for consistency with `forward` method.
            input_text_tokens = text_tokens.unsqueeze(0) if text_tokens.dim() == 1 else text_tokens
            
            # Call the main forward pass. For inference, `text_lengths` is typically not
            # explicitly needed as the input is usually padded to a fixed length or
            # the attention mechanism handles variable lengths implicitly.
            return self.forward(
                input_text_tokens,
                voice_id_tensor,
                language_id_tensor,
                text_lengths=None # No explicit text_lengths needed for single inference sample
            )

# --- Model Building and Management Utilities ---

def build_kokoro_model(config: dict) -> KokoroTTS:
    """
    Constructs an instance of the `KokoroTTS` model based on a provided configuration dictionary.
    This function simplifies model instantiation by mapping configuration parameters
    to the model's `__init__` arguments.

    Args:
        config (dict): A dictionary containing all necessary hyperparameters for the model.
                       Example keys and their default values are shown in `KOKORO_CONFIG`.

    Returns:
        KokoroTTS: An initialized instance of the `KokoroTTS` model.
    """
    print("Building KokoroTTS model with the following configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

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
    print("KokoroTTS model built successfully.")
    return model

def load_kokoro_checkpoint(model: nn.Module, checkpoint_path: str, device: str = 'cpu') -> tuple:
    """
    Loads a saved model checkpoint into a `KokoroTTS` model instance.
    This function is essential for resuming training or for deploying a pre-trained model.

    Args:
        model (nn.Module): The `KokoroTTS` model instance into which the state dictionary
                           from the checkpoint will be loaded.
        checkpoint_path (str): The file path to the saved `.pth` or `.pt` checkpoint file.
        device (str, optional): The device ('cpu' or 'cuda') to map the loaded tensors to.
                                This is important if the checkpoint was saved on a different device.
                                Defaults to 'cpu'.

    Returns:
        tuple: A tuple containing the `(epoch, step)` from the loaded checkpoint.
               These values are useful for resuming training from the exact point
               where the checkpoint was saved. Returns `(0, 0)` if these keys are
               not found in the checkpoint.
    
    Raises:
        FileNotFoundError: If the specified `checkpoint_path` does not exist.
        KeyError: If 'model_state_dict' is not found in the checkpoint.
    """
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Error: Checkpoint file not found at: '{checkpoint_path}'")

    print(f"Loading checkpoint from: '{checkpoint_path}' to device: '{device}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the model's state dictionary. `strict=True` (default) ensures all keys match.
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state dictionary loaded successfully.")
    
    # Retrieve epoch and step for resuming training. Use `.get()` with a default value
    # to handle cases where these keys might be missing in older checkpoints.
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    print(f"Checkpoint loaded: Epoch {epoch}, Step {step}.")
    return epoch, step

def save_kokoro_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int, checkpoint_path: str):
    """
    Saves the current state of the `KokoroTTS` model and its optimizer to a checkpoint file.
    This allows for training to be paused and resumed later, or for saving trained models.

    Args:
        model (nn.Module): The `KokoroTTS` model instance whose `state_dict` will be saved.
        optimizer (torch.optim.Optimizer): The optimizer instance whose `state_dict` will be saved.
        epoch (int): The current epoch number at the time of saving.
        step (int): The current training step number within the epoch.
        checkpoint_path (str): The full file path where the checkpoint will be saved (e.g., `checkpoints/model_ep01_step1000.pth`).
    """
    # Ensure the directory for the checkpoint exists. If not, create it.
    output_dir = osp.dirname(checkpoint_path)
    if output_dir: # Only create if path includes a directory
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a dictionary containing all necessary information to save.
    # `model_state_dict`: The learned parameters (weights, biases) of the model.
    # `optimizer_state_dict`: The state of the optimizer (e.g., learning rates, momentum buffers).
    # `epoch`, `step`: Metadata for resuming training.
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    
    # Save the dictionary to the specified path.
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Checkpoint saved to: '{checkpoint_path}' (Epoch: {epoch}, Step: {step})")

# --- Example Configuration for Kokoro-82M ---
# This dictionary defines the default hyperparameters for the Kokoro-82M model.
# It can be loaded from a YAML file in a real application for easier configuration management.
KOKORO_CONFIG = {
    'n_symbols': 256,         # Number of unique input text symbols (e.g., phoneme set size)
    'num_voices': 54,         # Number of distinct speaker identities
    'num_languages': 8,       # Number of languages supported (for VoicePacketEmbedding)
    'text_dim': 256,          # Dimensionality of text encoder outputs
    'voice_dim': 256,         # Dimensionality of voice/speaker embeddings
    'prosody_dim': 128,       # Dimensionality of prosody features output from predictor
    'mel_dim': 80,            # Dimensionality of mel-spectrogram features (e.g., 80 mel-bins)
    'hidden_dim': 512,        # Internal hidden dimensionality for decoder layers
    'num_decoder_layers': 6,  # Number of stacked decoder layers
    'dropout': 0.1            # Dropout probability for regularization
}


# --- Example Usage (Demonstrates Functionalities) ---
if __name__ == '__main__':
    print("\n--- Demonstrating Kokoro-82M Model Functionalities ---")

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Build the Model
    # We use the default KOKORO_CONFIG for demonstration.
    model = build_kokoro_model(KOKORO_CONFIG).to(device)
    print(f"Model successfully moved to {device}.")

    # 3. Instantiate the Loss Function
    loss_fn = KokoroLoss(mel_weight=1.0, prosody_weight=0.1)
    print("Loss function (KokoroLoss) instantiated.")

    # 4. Create a Dummy Optimizer (for checkpointing demonstration)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Dummy optimizer (Adam) instantiated.")

    # 5. Prepare Dummy Input Data for Training Forward Pass
    batch_size = 2
    max_text_len = 50
    # Simulate text tokens (e.g., phoneme IDs)
    dummy_text_tokens = torch.randint(0, KOKORO_CONFIG['n_symbols'], (batch_size, max_text_len), device=device)
    # Simulate actual lengths for padding
    dummy_text_lengths = torch.tensor([45, 30], dtype=torch.long, device=device)
    # Simulate voice IDs
    dummy_voice_ids = torch.randint(0, KOKORO_CONFIG['num_voices'], (batch_size,), device=device)
    # Simulate language IDs
    dummy_language_ids = torch.randint(0, KOKORO_CONFIG['num_languages'], (batch_size,), device=device)

    # Simulate target mel-spectrogram and prosody features
    # Mel-spectrogram length usually depends on text length and duration prediction.
    # For simplicity, let's assume a fixed ratio or a target length.
    # Here, we'll make it proportional to max_text_len for dummy data.
    mel_len = max_text_len * 5 # Example ratio
    dummy_target_mel = torch.randn(batch_size, mel_len, KOKORO_CONFIG['mel_dim'], device=device)
    dummy_target_f0 = torch.randn(batch_size, max_text_len, device=device)
    dummy_target_energy = torch.randn(batch_size, max_text_len, device=device)
    dummy_target_duration = torch.rand(batch_size, max_text_len, device=device) # Values between 0 and 1

    dummy_targets = {
        'mel': dummy_target_mel,
        'f0': dummy_target_f0,
        'energy': dummy_target_energy,
        'duration': dummy_target_duration
    }
    print("\nDummy training input and target data prepared.")

    # 6. Run a Training Forward Pass
    model.train() # Set model to training mode
    print("\nRunning model forward pass (training mode)...")
    predictions = model(dummy_text_tokens, dummy_voice_ids, dummy_language_ids, dummy_text_lengths)

    # Print shapes of predictions
    print("Training Predictions Shapes:")
    for k, v in predictions.items():
        print(f"  {k}: {v.shape}")

    # 7. Calculate Loss
    print("\nCalculating training loss...")
    losses = loss_fn(predictions, dummy_targets)
    print(f"  Total Loss: {losses['total_loss'].item():.4f}")
    print(f"  Mel Loss: {losses['mel_loss'].item():.4f}")
    print(f"  Prosody Loss: {losses['prosody_loss'].item():.4f}")

    # 8. Demonstrate Inference Pass (Single Sample)
    print("\nDemonstrating inference pass (single sample)...")
    model.eval() # Set model to evaluation mode
    
    # Prepare dummy input for inference (single sample)
    inference_text_tokens = torch.tensor([10, 20, 5, 30, 15, 25, 0, 0], dtype=torch.long, device=device) # Example sequence
    inference_voice_id = 0 # First voice
    inference_language_id = 0 # First language

    inference_output = model.inference(inference_text_tokens, inference_voice_id, inference_language_id)

    print("Inference Output Shapes (Batch size 1):")
    for k, v in inference_output.items():
        print(f"  {k}: {v.shape}")

    # 9. Demonstrate Checkpointing
    print("\nDemonstrating checkpointing functionality...")
    checkpoint_dir = "kokoro_checkpoints_demo"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = osp.join(checkpoint_dir, "kokoro_model_epoch_0_step_0.pth")

    # Save checkpoint
    save_kokoro_checkpoint(model, optimizer, epoch=0, step=0, checkpoint_path=checkpoint_path)

    # Load checkpoint into a new model instance to verify
    print("\nLoading checkpoint into a new model instance for verification...")
    new_model = build_kokoro_model(KOKORO_CONFIG).to(device)
    new_epoch, new_step = load_kokoro_checkpoint(new_model, checkpoint_path, device)
    print(f"New model loaded from checkpoint: Epoch {new_epoch}, Step {new_step}.")

    # Verify that loaded model produces same output (simple check)
    new_model.eval()
    with torch.no_grad():
        reloaded_inference_output = new_model.inference(inference_text_tokens, inference_voice_id, inference_language_id)
    
    # Compare a key output (e.g., mel_after)
    if torch.allclose(inference_output['mel_after'], reloaded_inference_output['mel_after'], atol=1e-6):
        print("Verification successful: Loaded model produces identical inference output.")
    else:
        print("Verification failed: Loaded model produces different inference output.")

    # Clean up dummy checkpoint directory
    import shutil
    shutil.rmtree(checkpoint_dir)
    print(f"\nCleaned up dummy checkpoint directory: '{checkpoint_dir}'")

    print("\n--- Kokoro-82M Model Functionalities Demonstration Complete ---")

import os
import os.path as osp
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

# Assuming these are external utilities and not part of the core Kokoro model definition
# from Utils.ASR.models import ASRCNN              
# from Utils.JDC.model import JDCNet
# from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator

from munch import Munch
import yaml

class LinearNorm(torch.nn.Module):
    """
    A linear layer with Xavier uniform weight initialization.
    This is a common practice to help with stable training in deep networks.
    """
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # Initialize weights using Xavier uniform initialization
        # 'linear' gain is suitable for linear activation functions
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class LayerNorm(nn.Module):
    """
    Layer Normalization module.
    Normalizes the input across the channel dimension.
    Equivalent to F.layer_norm but handles input tensor shape (B, C, T) by transposing.
    """
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # Transpose to (B, T, C) for F.layer_norm, then transpose back
        x = x.transpose(1, -1) # [B, T, C]
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1) # [B, C, T]

class KokoroTextEncoder(nn.Module):
    """
    Lightweight text encoder for Kokoro-82M.
    Processes text tokens into a sequence of rich feature representations.
    Uses CNNs for local feature extraction and an LSTM for sequential context.
    """
    def __init__(self, channels=256, kernel_size=5, depth=4, n_symbols=256, dropout=0.1):
        super().__init__()
        # Embedding layer to convert discrete text symbols into dense vectors
        self.embedding = nn.Embedding(n_symbols, channels)
        
        # Convolutional layers for extracting local features
        # Reduced depth (default 4) for efficiency as per Kokoro-82M design
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                # weight_norm applied to Conv1d for stable training
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels), # Custom LayerNorm for channel-first input
                nn.ReLU(inplace=True), # inplace=True saves a small amount of memory
                nn.Dropout(dropout),
            ))

        # Single bidirectional LSTM layer for capturing long-range dependencies
        # channels//2 because bidirectional LSTM outputs 2 * output_size
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)
        # Linear projection to ensure output dimension matches input channels
        self.output_proj = nn.Linear(channels, channels)

    def forward(self, x, input_lengths=None, mask=None):
        # x: [B, T_text] (batch_size, text_sequence_length)
        x = self.embedding(x)  # [B, T_text, channels]
        x = x.transpose(1, 2)  # [B, channels, T_text] - required for Conv1d

        # Apply mask to zero out padded regions before CNNs
        if mask is not None:
            # Unsqueeze mask to match [B, 1, T_text] for broadcasting with [B, channels, T_text]
            mask_expanded = mask.to(x.device).unsqueeze(1)
            x.masked_fill_(mask_expanded, 0.0)
        
        # Pass through CNN layers
        for c in self.cnn:
            x = c(x)
            # Re-apply mask after each CNN layer to ensure padded regions remain zero
            if mask is not None:
                x.masked_fill_(mask_expanded, 0.0)
                
        x = x.transpose(1, 2)  # [B, T_text, channels] - required for LSTM

        # Pack padded sequence for efficient LSTM processing
        if input_lengths is not None:
            # input_lengths must be on CPU for pack_padded_sequence
            # This is a CPU-GPU sync point, but often necessary for this operation.
            input_lengths_cpu = input_lengths.cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths_cpu, batch_first=True, enforce_sorted=False)

        # Flatten LSTM parameters for efficiency (e.g., when using DataParallel)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x) # x: [B, T_text, channels] (due to bidirectional output)
        
        # Pad packed sequence back to original tensor shape
        if input_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        x = self.output_proj(x) # Project back to original 'channels' dimension
        return x

class VoicePacketEmbedding(nn.Module):
    """
    Voice packet embedding for multi-speaker and multi-lingual synthesis.
    Combines voice identity and language information into a single embedding.
    """
    def __init__(self, num_voices=54, voice_dim=256):
        super().__init__()
        self.num_voices = num_voices
        self.voice_dim = voice_dim
        
        # Voice packet embeddings: Maps voice_id to a dense vector
        self.voice_embeddings = nn.Embedding(num_voices, voice_dim)
        
        # Language embeddings: Maps language_id to a dense vector
        # voice_dim // 4 is a design choice for the language embedding size
        self.language_embeddings = nn.Embedding(8, voice_dim // 4) # Assuming 8 languages
        
        # Projection layer to combine voice and language embeddings
        # Input dimension is voice_dim + (voice_dim // 4)
        self.proj = nn.Linear(voice_dim + voice_dim // 4, voice_dim)
        
    def forward(self, voice_id, language_id=None):
        # voice_id: [B] or scalar
        voice_emb = self.voice_embeddings(voice_id) # [B, voice_dim]
        
        if language_id is not None:
            # language_id: [B] or scalar
            lang_emb = self.language_embeddings(language_id) # [B, voice_dim // 4] or [voice_dim // 4]

            # Ensure lang_emb has the same batch dimension as voice_emb for concatenation
            # If lang_emb is a scalar embedding (e.g., for inference with a single language_id),
            # expand it to match the batch size of voice_emb.
            if lang_emb.dim() == 1: # Case where language_id was a scalar and embedding output is [D]
                lang_emb = lang_emb.unsqueeze(0) # Make it [1, D]
            
            # If voice_emb is [B, D] and lang_emb is [B, D_lang], simply concatenate
            # This handles both batch and single-sample cases correctly.
            combined = torch.cat([voice_emb, lang_emb], dim=-1)
            return self.proj(combined)
            
        return voice_emb # If no language_id is provided, return only voice embedding

class KokoroProsodyPredictor(nn.Module):
    """
    Lightweight prosody predictor for natural intonation.
    Predicts F0 (fundamental frequency), energy, and duration based on
    text features and speaker/language embeddings.
    """
    def __init__(self, text_dim=256, voice_dim=256, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        # Linear projections to align dimensions before concatenation
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.voice_proj = nn.Linear(voice_dim, hidden_dim)
        
        # Simplified prosody prediction network (MLP-like)
        # Input to first linear layer is hidden_dim * 2 (from concatenated text_proj and voice_proj)
        self.prosody_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Separate heads for different prosodic features
        # Each head outputs a single scalar per time step
        self.f0_head = nn.Linear(hidden_dim // 2, 1)  # F0 prediction
        self.energy_head = nn.Linear(hidden_dim // 2, 1)  # Energy prediction
        # Sigmoid activation for duration to keep values between 0 and 1 (e.g., relative duration)
        self.duration_head = nn.Linear(hidden_dim // 2, 1)  # Duration prediction
        
    def forward(self, text_features, voice_embedding):
        # text_features: [B, T_text, text_dim]
        # voice_embedding: [B, voice_dim] (or [voice_dim] for inference)
        
        # Project inputs to common hidden_dim
        text_proj = self.text_proj(text_features)  # [B, T_text, hidden_dim]
        voice_proj = self.voice_proj(voice_embedding)  # [B, hidden_dim]
        
        # Expand voice embedding to match text sequence length for element-wise combination
        # This broadcasting is efficient as it doesn't create a new large tensor in memory
        if voice_proj.dim() == 2:
            voice_proj = voice_proj.unsqueeze(1).expand(-1, text_proj.size(1), -1)
            
        # Combine text and voice features by concatenation
        combined = torch.cat([text_proj, voice_proj], dim=-1)  # [B, T_text, hidden_dim * 2]
        
        # Predict prosodic features through the network
        prosody_features = self.prosody_net(combined) # [B, T_text, hidden_dim // 2]
        
        # Apply separate heads and squeeze the last dimension (which is 1)
        f0 = self.f0_head(prosody_features).squeeze(-1)  # [B, T_text]
        energy = self.energy_head(prosody_features).squeeze(-1)  # [B, T_text]
        duration = torch.sigmoid(self.duration_head(prosody_features)).squeeze(-1)  # [B, T_text]
        
        return {
            'f0': f0,
            'energy': energy,
            'duration': duration,
            'prosody_features': prosody_features # Return for decoder input
        }

class KokoroDecoder(nn.Module):
    """
    Lightweight decoder-only model for mel-spectrogram generation.
    Takes text, voice, and prosody features to produce mel-spectrograms.
    Uses a simplified transformer-like architecture for decoding.
    """
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
        
        # Input projection: Combines all input features into the decoder's hidden_dim
        # The prosody_features from ProsodyPredictor have hidden_dim // 2, so prosody_dim here
        # refers to the conceptual dimension, but the actual input is hidden_dim // 2.
        # It should be text_dim + voice_dim + (prosody_predictor_output_dim)
        # From KokoroProsodyPredictor, prosody_features has hidden_dim // 2, which is `prosody_dim`
        # if `prosody_predictor`'s `hidden_dim` is `prosody_dim * 2`.
        self.input_proj = nn.Linear(text_dim + voice_dim + prosody_dim, hidden_dim)
        
        # Decoder layers: Stack of simplified transformer blocks
        self.decoder_layers = nn.ModuleList([
            KokoroDecoderLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # Output projection to mel-spectrogram
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, mel_dim) # Final projection to mel_dim
        )
        
        # Postnet for mel-spectrogram refinement
        # hidden_dim // 4 is a design choice for Postnet's internal hidden_dim
        self.postnet = KokoroPostnet(mel_dim, hidden_dim // 4)
        
    def forward(self, text_features, voice_embedding, prosody_features, lengths=None):
        # text_features: [B, T, text_dim]
        # voice_embedding: [B, voice_dim]
        # prosody_features: [B, T, prosody_dim] (where prosody_dim is hidden_dim // 2 from predictor)
        
        batch_size, seq_len, _ = text_features.shape
        
        # Expand voice embedding to match sequence length for concatenation
        # This is an efficient broadcasting operation
        if voice_embedding.dim() == 2:
            voice_embedding = voice_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            
        # Combine all features into a single input for the decoder
        combined_features = torch.cat([text_features, voice_embedding, prosody_features], dim=-1)
        
        # Project combined features to the decoder's hidden dimension
        x = self.input_proj(combined_features)  # [B, T, hidden_dim]
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, lengths) # lengths are used to create attention mask
            
        # Generate initial mel-spectrogram (mel_before)
        mel_before = self.output_proj(x)  # [B, T, mel_dim]
        # Apply Postnet for refinement: Postnet output is added as a residual
        mel_after = self.postnet(mel_before) + mel_before
        
        return {
            'mel_before': mel_before,
            'mel_after': mel_after
        }

class KokoroDecoderLayer(nn.Module):
    """
    Single decoder layer with self-attention and feed-forward network.
    Forms the building block of the KokoroDecoder.
    """
    def __init__(self, hidden_dim, dropout=0.1, num_heads=8):
        super().__init__()
        
        # Self-attention mechanism
        # batch_first=True means input/output tensors are (batch, sequence, feature)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim) # LayerNorm after attention
        
        # Feed-forward network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), # Expand to 2x hidden_dim
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), # Project back to hidden_dim
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim) # LayerNorm after FFN
        
        self.dropout = nn.Dropout(dropout) # Dropout for residual connections
        
    def forward(self, x, lengths=None):
        # x: [B, T, hidden_dim]
        
        # Create attention mask to prevent attention to padding tokens
        attn_mask = None
        if lengths is not None:
            max_len = x.size(1)
            # attn_mask is a boolean tensor of shape [B, T_query, T_key]
            # True indicates positions that should be masked (i.e., ignored)
            # Here, it masks positions beyond the actual sequence length.
            # Example: lengths = [3, 5], max_len = 5
            # torch.arange(5) = [0, 1, 2, 3, 4]
            # lengths.unsqueeze(1) = [[3], [5]]
            # result:
            # [[F, F, F, T, T],
            #  [F, F, F, F, F]]
            attn_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Self-attention block with residual connection and LayerNorm
        residual = x
        # key_padding_mask is used to mask out padded positions in the key/value sequences
        x_attn, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(residual + self.dropout(x_attn))
        
        # Feed-forward block with residual connection and LayerNorm
        residual = x
        x_ffn = self.ffn(x)
        x = self.norm2(residual + x_ffn)
        
        return x

class KokoroPostnet(nn.Module):
    """
    Postnet for mel-spectrogram refinement.
    A stack of 1D convolutional layers used to predict a residual
    that is added to the mel-spectrogram generated by the decoder.
    This helps in refining the spectral details.
    """
    def __init__(self, mel_dim=80, hidden_dim=128, num_layers=5, kernel_size=5):
        super().__init__()
        
        padding = (kernel_size - 1) // 2 # Calculate padding to maintain sequence length
        
        self.convs = nn.ModuleList()
        
        # First convolutional layer: mel_dim -> hidden_dim
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(mel_dim, hidden_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(), # Tanh activation is common in Postnets
                nn.Dropout(0.5)
            )
        )
        
        # Intermediate convolutional layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2): # num_layers - 2 because first and last are separate
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5)
                )
            )
            
        # Last convolutional layer: hidden_dim -> mel_dim
        # No Tanh activation here, as it's a residual prediction
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(hidden_dim, mel_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(0.5)
            )
        )
        
    def forward(self, x):
        # x: [B, T_mel, mel_dim] (batch_size, mel_sequence_length, mel_features)
        
        # Transpose to [B, mel_dim, T_mel] for Conv1d operations
        x = x.transpose(1, 2)
        
        # Pass through all convolutional blocks
        for conv in self.convs:
            x = conv(x)
            
        # Transpose back to [B, T_mel, mel_dim] for consistency with decoder output
        x = x.transpose(1, 2)
        return x

class KokoroTTS(nn.Module):
    """
    Complete Kokoro-82M Text-to-Speech model.
    Integrates all sub-modules to perform end-to-end speech synthesis.
    """
    def __init__(self, 
                 n_symbols=256,
                 num_voices=54,
                 num_languages=8,
                 text_dim=256,
                 voice_dim=256,
                 prosody_dim=128, # This is the output dimension of prosody_features from predictor
                 mel_dim=80,
                 hidden_dim=512,
                 num_decoder_layers=6,
                 dropout=0.1):
        super().__init__()
        
        # Initialize core components of the TTS model
        self.text_encoder = KokoroTextEncoder(
            channels=text_dim, 
            n_symbols=n_symbols,
            dropout=dropout
        )
        
        self.voice_embedding = VoicePacketEmbedding(
            num_voices=num_voices,
            voice_dim=voice_dim
        )
        
        # Note: prosody_predictor's hidden_dim is set to prosody_dim * 2
        # Its output 'prosody_features' will have dimension (prosody_dim * 2) // 2 = prosody_dim
        self.prosody_predictor = KokoroProsodyPredictor(
            text_dim=text_dim,
            voice_dim=voice_dim,
            hidden_dim=prosody_dim * 2, # Internal hidden dimension for prosody predictor
            dropout=dropout
        )
        
        self.decoder = KokoroDecoder(
            text_dim=text_dim,
            voice_dim=voice_dim,
            prosody_dim=prosody_dim, # Matches the output dimension of prosody_predictor's features
            mel_dim=mel_dim,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            dropout=dropout
        )
        
    def forward(self, text_tokens, voice_ids, language_ids=None, text_lengths=None):
        """
        Forward pass for training the Kokoro-82M model.
        
        Args:
            text_tokens (torch.Tensor): Batch of text token sequences. Shape: [B, T_text]
            voice_ids (torch.Tensor): Batch of voice IDs. Shape: [B]
            language_ids (torch.Tensor, optional): Batch of language IDs. Shape: [B]. Defaults to None.
            text_lengths (torch.Tensor, optional): Lengths of text sequences. Shape: [B]. Defaults to None.
        
        Returns:
            dict: Contains predicted mel-spectrograms (before and after Postnet),
                  and predicted prosodic features (F0, energy, duration).
        """
        # Encode text tokens into a sequence of features
        text_features = self.text_encoder(text_tokens, text_lengths) # [B, T_text, text_dim]
        
        # Get voice (and optionally language) embeddings
        voice_embeddings = self.voice_embedding(voice_ids, language_ids) # [B, voice_dim]
        
        # Predict prosodic features (F0, energy, duration) and their intermediate features
        prosody_output = self.prosody_predictor(text_features, voice_embeddings)
        
        # Generate mel-spectrograms using text, voice, and prosody features
        decoder_output = self.decoder(
            text_features, 
            voice_embeddings, 
            prosody_output['prosody_features'], # Use the intermediate prosody features
            text_lengths # Pass text_lengths for attention masking in the decoder
        )
        
        return {
            'mel_before': decoder_output['mel_before'],
            'mel_after': decoder_output['mel_after'],
            'f0': prosody_output['f0'],
            'energy': prosody_output['energy'],
            'duration': prosody_output['duration']
        }
    
    def inference(self, text_tokens, voice_id, language_id=None):
        """
        Inference mode for single sample generation.
        Sets the model to evaluation mode and disables gradient computation.
        
        Args:
            text_tokens (torch.Tensor): Text token sequence. Shape: [T_text] or [1, T_text]
            voice_id (int or torch.Tensor): Single voice ID.
            language_id (int or torch.Tensor, optional): Single language ID. Defaults to None.
            
        Returns:
            dict: Contains predicted mel-spectrograms and prosodic features for the single sample.
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient computation for inference
            # Ensure voice_id and language_id are tensors with batch dimension
            if isinstance(voice_id, int):
                voice_id = torch.tensor([voice_id], device=text_tokens.device)
            if language_id is not None and isinstance(language_id, int):
                language_id = torch.tensor([language_id], device=text_tokens.device)
                
            # Ensure text_tokens has a batch dimension for consistency with forward pass
            input_text_tokens = text_tokens.unsqueeze(0) if text_tokens.dim() == 1 else text_tokens
            
            # No text_lengths needed for inference if input_text_tokens is already padded/unpadded correctly
            # and attention masking handles it. For simplicity, we omit it here, assuming
            # the attention mechanism can handle variable lengths or fixed max lengths.
            # If dynamic padding is critical for inference, text_lengths should be passed.
            return self.forward(
                input_text_tokens,
                voice_id,
                language_id,
                text_lengths=None # For inference, typically full sequence is passed or handled internally
            )

def build_kokoro_model(config):
    """
    Build Kokoro-82M model with given configuration.
    
    Args:
        config (dict): Configuration dictionary containing model hyperparameters.
        
    Returns:
        KokoroTTS: An instance of the KokoroTTS model.
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

# Loss functions for training
class KokoroLoss(nn.Module):
    """
    Combined loss function for Kokoro-82M training.
    Calculates mel-spectrogram reconstruction loss and prosody prediction loss.
    """
    def __init__(self, mel_weight=1.0, prosody_weight=0.1):
        super().__init__()
        self.mel_weight = mel_weight
        self.prosody_weight = prosody_weight
        
        self.l1_loss = nn.L1Loss() # For mel-spectrogram reconstruction
        self.mse_loss = nn.MSELoss() # For prosody prediction (F0, energy, duration)
        
    def forward(self, predictions, targets):
        """
        Calculates the total loss.
        
        Args:
            predictions (dict): Dictionary of model predictions (mel_before, mel_after, f0, energy, duration).
            targets (dict): Dictionary of target values (mel, f0, energy, duration).
            
        Returns:
            dict: Contains total loss, mel loss, and prosody loss components.
        """
        # Mel-spectrogram reconstruction loss: L1 loss for both before and after Postnet
        mel_loss_before = self.l1_loss(predictions['mel_before'], targets['mel'])
        mel_loss_after = self.l1_loss(predictions['mel_after'], targets['mel'])
        mel_loss = mel_loss_before + mel_loss_after
        
        # Prosody losses: MSE loss for F0, energy, and duration
        prosody_loss = 0
        # Added shape checks to prevent errors if target keys exist but shapes don't match
        if 'f0' in targets and predictions['f0'].shape == targets['f0'].shape:
            prosody_loss += self.mse_loss(predictions['f0'], targets['f0'])
        if 'energy' in targets and predictions['energy'].shape == targets['energy'].shape:
            prosody_loss += self.mse_loss(predictions['energy'], targets['energy'])
        if 'duration' in targets and predictions['duration'].shape == targets['duration'].shape:
            prosody_loss += self.mse_loss(predictions['duration'], targets['duration'])
            
        # Total loss is a weighted sum
        total_loss = self.mel_weight * mel_loss + self.prosody_weight * prosody_loss
        
        return {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'prosody_loss': prosody_loss
        }

# Utility functions
def load_kokoro_checkpoint(model, checkpoint_path, device='cpu'):
    """
    Load Kokoro model checkpoint.
    
    Args:
        model (nn.Module): The model instance to load state into.
        checkpoint_path (str): Path to the checkpoint file.
        device (str): Device to map the checkpoint to ('cpu' or 'cuda').
        
    Returns:
        tuple: (epoch, step) from the loaded checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Return epoch and step for resuming training
    return checkpoint.get('epoch', 0), checkpoint.get('step', 0)

def save_kokoro_checkpoint(model, optimizer, epoch, step, checkpoint_path):
    """
    Save Kokoro model checkpoint.
    
    Args:
        model (nn.Module): The model instance to save.
        optimizer (torch.optim.Optimizer): The optimizer instance to save.
        epoch (int): Current epoch number.
        step (int): Current training step number.
        checkpoint_path (str): Path to save the checkpoint file.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }, checkpoint_path)

# Example configuration
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

import os
import os.path as osp
import json
import math
import sys
from typing import Optional, Dict, Tuple, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

# Add iSTFTNet path to system path
ISTFTNET_PATH = "/home/humair/iSTFTNet-pytorch"
if ISTFTNET_PATH not in sys.path:
    sys.path.append(ISTFTNET_PATH)

# Import iSTFTNet components
try:
    from models import iSTFTNet
    from utils import get_mel
    import env
    ISTFTNET_AVAILABLE = True
    print("iSTFTNet successfully imported")
except ImportError as e:
    print(f"Warning: Could not import iSTFTNet: {e}")
    print("Falling back to simple vocoder")
    ISTFTNET_AVAILABLE = False

class LinearNorm(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = 'linear'):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

class LayerNorm(nn.Module):
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

class KokoroTextEncoder(nn.Module):
    def __init__(self, channels: int = 256, kernel_size: int = 5, depth: int = 4,
                 n_symbols: int = 256, dropout: float = 0.1):
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

        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, input_lengths: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)
        x = x.transpose(1, 2)

        if mask is not None:
            mask_expanded = mask.to(x.device).unsqueeze(1)
            x.masked_fill_(mask_expanded, 0.0)
        
        for c in self.cnn:
            x = c(x)
            if mask is not None:
                x.masked_fill_(mask_expanded, 0.0)
                
        x = x.transpose(1, 2)

        if input_lengths is not None:
            input_lengths_cpu = input_lengths.cpu()
            x = nn.utils.rnn.pack_padded_sequence(x, input_lengths_cpu, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        
        if input_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        x = self.output_proj(x)
        return x

class VoicePacketEmbedding(nn.Module):
    def __init__(self, num_voices: int = 54, voice_dim: int = 256, num_languages: int = 8):
        super().__init__()
        self.num_voices = num_voices
        self.voice_dim = voice_dim
        self.voice_embeddings = nn.Embedding(num_voices, voice_dim)
        self.language_embeddings = nn.Embedding(num_languages, voice_dim // 4)
        self.proj = nn.Linear(voice_dim + voice_dim // 4, voice_dim)
        
    def forward(self, voice_id: torch.Tensor, language_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        voice_emb = self.voice_embeddings(voice_id)

        if language_id is not None:
            lang_emb = self.language_embeddings(language_id)

            if voice_emb.dim() == 1 and lang_emb.dim() == 1:
                voice_emb = voice_emb.unsqueeze(0)
                lang_emb = lang_emb.unsqueeze(0)
            elif voice_emb.dim() == 1 and lang_emb.dim() == 2:
                voice_emb = voice_emb.unsqueeze(0)

            combined = torch.cat([voice_emb, lang_emb], dim=-1)
            return self.proj(combined)
            
        return voice_emb

class KokoroProsodyPredictor(nn.Module):
    def __init__(self, text_dim: int = 256, voice_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
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
        
    def forward(self, text_features: torch.Tensor, voice_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_proj = self.text_proj(text_features)
        voice_proj = self.voice_proj(voice_embedding)
        
        if voice_proj.dim() == 0:
            voice_proj = voice_proj.unsqueeze(0)
        if voice_proj.dim() == 1:
            voice_proj = voice_proj.unsqueeze(0)
        
        voice_proj_expanded = voice_proj.unsqueeze(1).expand(-1, text_proj.size(1), -1)
            
        combined = torch.cat([text_proj, voice_proj_expanded], dim=-1)
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

class KokoroDecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, num_heads: int = 8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
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
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    def __init__(self, mel_dim: int = 80, hidden_dim: int = 128, num_layers: int = 5, kernel_size: int = 5):
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        self.convs = nn.ModuleList()
        
        self.convs.append(nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.5)
        ))
        
        for _ in range(num_layers - 2):
            self.convs.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.Tanh(),
                nn.Dropout(0.5)
            ))
            
        self.convs.append(nn.Sequential(
            nn.Conv1d(hidden_dim, mel_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(mel_dim),
            nn.Dropout(0.5)
        ))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        return x

class KokoroDecoder(nn.Module):
    def __init__(self, text_dim: int = 256, voice_dim: int = 256, prosody_dim: int = 128,
                 mel_dim: int = 80, hidden_dim: int = 512, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(text_dim + voice_dim + prosody_dim, hidden_dim)
        self.decoder_layers = nn.ModuleList([KokoroDecoderLayer(hidden_dim, dropout) for _ in range(num_layers)])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, mel_dim)
        )
        
        self.postnet = KokoroPostnet(mel_dim, hidden_dim // 4)
        
    def forward(self, text_features: torch.Tensor, voice_embedding: torch.Tensor, 
                prosody_features: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = text_features.shape
        
        if voice_embedding.dim() == 0:
            voice_embedding = voice_embedding.unsqueeze(0)
        if voice_embedding.dim() == 1:
            voice_embedding = voice_embedding.unsqueeze(0)
        voice_embedding_expanded = voice_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            
        combined_features = torch.cat([text_features, voice_embedding_expanded, prosody_features], dim=-1)
        x = self.input_proj(combined_features)
        
        for layer in self.decoder_layers:
            x = layer(x, lengths)
            
        mel_before = self.output_proj(x)
        mel_after = self.postnet(mel_before) + mel_before
        
        return {'mel_before': mel_before, 'mel_after': mel_after}

class SimpleVocoder(nn.Module):
    """Fallback simple vocoder when iSTFTNet is not available"""
    def __init__(self, sample_rate: int = 24000):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        batch_size, mel_frames, mel_dim = mel_spectrogram.shape
        audio_len = int(mel_frames * (self.sample_rate / 80))
        t = torch.linspace(0, audio_len / self.sample_rate, audio_len, device=mel_spectrogram.device)
        frequency = 440
        amplitude = 0.3
        audio = amplitude * torch.sin(2 * np.pi * frequency * t)
        
        return audio.unsqueeze(0).expand(batch_size, -1)

class iSTFTNetVocoder(nn.Module):
    """Wrapper for iSTFTNet vocoder"""
    def __init__(self, checkpoint_path: str = None, config_path: str = None, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
        # Default paths
        if checkpoint_path is None:
            checkpoint_path = osp.join(ISTFTNET_PATH, "istftnet.pth")
        if config_path is None:
            config_path = osp.join(ISTFTNET_PATH, "config_v1.json")
            
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Initialize model
        self.model = iSTFTNet(self.config)
        
        # Load checkpoint if available
        if osp.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'generator' in checkpoint:
                self.model.load_state_dict(checkpoint['generator'])
                print(f"Loaded iSTFTNet checkpoint from {checkpoint_path}")
            else:
                print(f"Warning: No 'generator' key found in checkpoint {checkpoint_path}")
        else:
            print(f"Warning: iSTFTNet checkpoint not found at {checkpoint_path}")
            print("Using randomly initialized iSTFTNet (audio quality will be poor)")
            
        self.model.to(device)
        self.model.eval()
        
    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to audio waveform
        Args:
            mel_spectrogram: [batch_size, mel_frames, mel_dim]
        Returns:
            audio: [batch_size, audio_samples]
        """
        # Transpose to [batch_size, mel_dim, mel_frames] for iSTFTNet
        mel = mel_spectrogram.transpose(1, 2)
        
        with torch.no_grad():
            audio = self.model(mel)
            
        return audio.squeeze(1)  # Remove channel dimension

class KokoroTTS(nn.Module):
    def __init__(self, n_symbols: int = 256, num_voices: int = 54, num_languages: int = 8,
                 text_dim: int = 256, voice_dim: int = 256, prosody_dim: int = 128,
                 mel_dim: int = 80, hidden_dim: int = 512, num_decoder_layers: int = 6, 
                 dropout: float = 0.1, use_istftnet: bool = True, vocoder_checkpoint: str = None):
        super().__init__()
        
        self.text_encoder = KokoroTextEncoder(
            channels=text_dim, n_symbols=n_symbols, dropout=dropout)
        
        self.voice_embedding = VoicePacketEmbedding(
            num_voices=num_voices, voice_dim=voice_dim, num_languages=num_languages)
        
        self.prosody_predictor = KokoroProsodyPredictor(
            text_dim=text_dim, voice_dim=voice_dim, hidden_dim=prosody_dim * 2, dropout=dropout)
        
        self.decoder = KokoroDecoder(
            text_dim=text_dim, voice_dim=voice_dim, prosody_dim=prosody_dim,
            mel_dim=mel_dim, hidden_dim=hidden_dim, num_layers=num_decoder_layers, dropout=dropout)
        
        # Initialize vocoder
        if use_istftnet and ISTFTNET_AVAILABLE:
            try:
                device = next(self.parameters()).device if list(self.parameters()) else 'cpu'
                self.vocoder = iSTFTNetVocoder(
                    checkpoint_path=vocoder_checkpoint,
                    device=str(device)
                )
                print("Using iSTFTNet vocoder")
            except Exception as e:
                print(f"Failed to initialize iSTFTNet: {e}")
                print("Falling back to simple vocoder")
                self.vocoder = SimpleVocoder(sample_rate=24000)
        else:
            self.vocoder = SimpleVocoder(sample_rate=24000)
            print("Using simple vocoder")
        
    def forward(self, text_tokens: torch.Tensor, voice_ids: torch.Tensor,
                language_ids: Optional[torch.Tensor] = None,
                text_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        text_features = self.text_encoder(text_tokens, text_lengths)
        voice_embeddings = self.voice_embedding(voice_ids, language_ids)
        prosody_output = self.prosody_predictor(text_features, voice_embeddings)
        
        decoder_output = self.decoder(
            text_features, voice_embeddings, prosody_output['prosody_features'], text_lengths)
        
        return {
            'mel_before': decoder_output['mel_before'],
            'mel_after': decoder_output['mel_after'],
            'f0': prosody_output['f0'],
            'energy': prosody_output['energy'],
            'duration': prosody_output['duration']
        }
    
    def inference(self, text_tokens: torch.Tensor, voice_id: Union[int, torch.Tensor],
                  language_id: Optional[Union[int, torch.Tensor]] = None) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        self.eval()
        with torch.no_grad():
            device = text_tokens.device

            if isinstance(voice_id, int):
                voice_id_tensor = torch.tensor([voice_id], device=device)
            elif voice_id.dim() == 0:
                voice_id_tensor = voice_id.unsqueeze(0).to(device)
            else:
                voice_id_tensor = voice_id.to(device)

            language_id_tensor = None
            if language_id is not None:
                if isinstance(language_id, int):
                    language_id_tensor = torch.tensor([language_id], device=device)
                elif language_id.dim() == 0:
                    language_id_tensor = language_id.unsqueeze(0).to(device)
                else:
                    language_id_tensor = language_id.to(device)
                
            input_text_tokens = text_tokens.unsqueeze(0) if text_tokens.dim() == 1 else text_tokens
            
            acoustic_output = self.forward(input_text_tokens, voice_id_tensor, language_id_tensor, text_lengths=None)
            predicted_mel = acoustic_output['mel_after']
            
            # Move vocoder to same device as mel spectrogram
            if hasattr(self.vocoder, 'model'):
                self.vocoder.model.to(predicted_mel.device)
            
            audio_waveform = self.vocoder(predicted_mel)
            audio_np = audio_waveform.squeeze(0).cpu().numpy()
            
            return audio_np, acoustic_output

class KokoroLoss(nn.Module):
    def __init__(self, mel_weight: float = 1.0, prosody_weight: float = 0.1):
        super().__init__()
        self.mel_weight = mel_weight
        self.prosody_weight = prosody_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mel_loss_before = self.l1_loss(predictions['mel_before'], targets['mel'])
        mel_loss_after = self.l1_loss(predictions['mel_after'], targets['mel'])
        mel_loss = mel_loss_before + mel_loss_after
        
        prosody_loss = torch.tensor(0.0, device=predictions['f0'].device)
        
        if 'f0' in targets and predictions['f0'].shape == targets['f0'].shape:
            prosody_loss += self.mse_loss(predictions['f0'], targets['f0'])
        if 'energy' in targets and predictions['energy'].shape == targets['energy'].shape:
            prosody_loss += self.mse_loss(predictions['energy'], targets['energy'])
        if 'duration' in targets and predictions['duration'].shape == targets['duration'].shape:
            prosody_loss += self.mse_loss(predictions['duration'], targets['duration'])
            
        total_loss = self.mel_weight * mel_loss + self.prosody_weight * prosody_loss
        
        return {'total_loss': total_loss, 'mel_loss': mel_loss, 'prosody_loss': prosody_loss}

def build_kokoro_model(config: Dict, use_istftnet: bool = True, vocoder_checkpoint: str = None) -> KokoroTTS:
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
        dropout=config.get('dropout', 0.1),
        use_istftnet=use_istftnet,
        vocoder_checkpoint=vocoder_checkpoint
    )
    return model

def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = 'cpu') -> Tuple[int, int]:
    if not osp.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('step', 0)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, step: int, checkpoint_path: str):
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
    'n_symbols': 128,
    'num_voices': 10,
    'num_languages': 1,
    'text_dim': 256,
    'voice_dim': 256,
    'prosody_dim': 128,
    'mel_dim': 80,
    'hidden_dim': 512,
    'num_decoder_layers': 6,
    'dropout': 0.1,
    'mel_frames_per_token': 5
}

char_set = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'"
char_to_id = {char: i + 1 for i, char in enumerate(char_set)}
char_to_id['<pad>'] = 0

KOKORO_CONFIG['n_symbols'] = len(char_to_id)

def simple_tokenize(text: str, char_map: Dict[str, int]) -> torch.Tensor:
    tokens = [char_map.get(char, char_map[' ']) for char in text]
    return torch.tensor(tokens, dtype=torch.long)

class TTSDataset(Dataset):
    def __init__(self, metadata_path: str, audio_dir: str, config: Dict):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.audio_dir = audio_dir
        self.config = config
        self.text_keys = sorted(list(self.metadata.keys()))

        self.voice_id_map = {}
        for key in self.text_keys:
            try:
                num_str = key.split('_')[1].split('.')[0]
                voice_id = int(num_str)
                self.voice_id_map[key] = voice_id
            except (IndexError, ValueError):
                self.voice_id_map[key] = 0

        self.language_ids = {key: 0 for key in self.text_keys}

        max_assigned_voice_id = max(self.voice_id_map.values())
        if max_assigned_voice_id >= self.config['num_voices']:
            print(f"Warning: Max voice ID ({max_assigned_voice_id}) >= num_voices ({self.config['num_voices']})")

    def __len__(self) -> int:
        return len(self.text_keys)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        filename = self.text_keys[idx]
        text = self.metadata[filename]

        tokenized_text = simple_tokenize(text, char_to_id)
        text_length = len(tokenized_text)
        mel_length = text_length
        
        mel_target = torch.randn(mel_length, self.config['mel_dim'])
        f0_target = torch.randn(mel_length)
        energy_target = torch.randn(mel_length)
        duration_target = torch.rand(text_length) * (self.config['mel_frames_per_token'] * 2) + 1

        voice_id = self.voice_id_map[filename]
        language_id = self.language_ids[filename]

        return {
            'text_tokens': tokenized_text,
            'text_length': text_length,
            'voice_id': voice_id,
            'language_id': language_id,
            'mel_target': mel_target,
            'f0_target': f0_target,
            'energy_target': energy_target,
            'duration_target': duration_target
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_text_len = max(item['text_length'] for item in batch)
    max_mel_len = max_text_len

    padded_text_tokens = torch.full((len(batch), max_text_len), char_to_id['<pad>'], dtype=torch.long)
    text_lengths = torch.tensor([item['text_length'] for item in batch], dtype=torch.long)

    padded_mel_target = torch.zeros((len(batch), max_mel_len, KOKORO_CONFIG['mel_dim']))
    padded_f0_target = torch.zeros((len(batch), max_mel_len))
    padded_energy_target = torch.zeros((len(batch), max_mel_len))
    padded_duration_target = torch.zeros((len(batch), max_text_len))

    voice_ids = torch.tensor([item['voice_id'] for item in batch], dtype=torch.long)
    language_ids = torch.tensor([item['language_id'] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        padded_text_tokens[i, :item['text_length']] = item['text_tokens']
        padded_mel_target[i, :item['mel_target'].shape[0]] = item['mel_target']
        padded_f0_target[i, :item['f0_target'].shape[0]] = item['f0_target']
        padded_energy_target[i, :item['energy_target'].shape[0]] = item['energy_target']
        padded_duration_target[i, :item['duration_target'].shape[0]] = item['duration_target']

    return {
        'text_tokens': padded_text_tokens,
        'text_lengths': text_lengths,
        'voice_ids': voice_ids,
        'language_ids': language_ids,
        'mel': padded_mel_target,
        'f0': padded_f0_target,
        'energy': padded_energy_target,
        'duration': padded_duration_target

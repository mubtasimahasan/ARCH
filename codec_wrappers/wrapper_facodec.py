import os
import torch
import torchaudio
import numpy as np
from arch_eval import Model

class FACodecWrapper(Model):
    def __init__(self, encoder, decoder, device="cuda", max_length=None):
        """
        :param encoder: Pre-loaded FACodecEncoder
        :param decoder: Pre-loaded FACodecDecoder
        :param device: Device to run on
        :param max_length: Optional truncation length in samples
        """
        super().__init__(model=None)
        self.encoder = encoder.to(device).eval()
        self.decoder = decoder.to(device).eval()
        self.device = torch.device(device)
        self.sample_rate = 16000
        self.max_length = max_length

    def get_sampling_rate(self):
        return self.sample_rate

    def get_classification_embedding_size(self):
        return self.get_token_embedding_size()

    def get_token_embedding_size(self):
        # hardcoded for simplicity
        return 256

    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            wav = audio.float()
        else:
            wav = torch.from_numpy(audio).float()

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # (1, T)
        elif wav.shape[0] > 1:
            wav = wav[:1, :]  # mono

        if self.max_length:
            wav = wav[:, :self.max_length]

        return wav.unsqueeze(0).to(self.device)  # (1, 1, T)

    def get_sequence_embeddings(self, audio: np.ndarray, **kwargs):
        wav = self._preprocess_audio(audio)

        with torch.no_grad():
            enc_out = self.encoder(wav)
            _, _, _, quantized, _ = self.decoder(enc_out, eval_vq=False, vq=True)
            
            quantized = quantized[0]  # (n_q, D, T)
            quantized = quantized.permute(2, 0, 1).flatten(1)  # (T, n_q * D)

        return quantized.cpu()

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        seq_emb = self.get_sequence_embeddings(audio)
        return seq_emb.mean(dim=0)  # (D,)

    def get_token_embeddings(self, audio: np.ndarray, **kwargs):
        return self.get_sequence_embeddings(audio)

    def get_embedding_layer(self):
        return self.embedding_dim

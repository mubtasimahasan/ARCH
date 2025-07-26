import os
import torch
import numpy as np
import torchaudio
from arch_eval import Model

# DAC import
import dac
from audiotools import AudioSignal

class DACWrapper(Model):
    def __init__(self, device="cuda", n_q=8, max_length=None):
        super().__init__(model=None)
        self.device = torch.device(device)
        self.sample_rate = 16000
        self.n_q = n_q
        self.max_length = max_length

        # Load DAC
        model_path = dac.utils.download(model_type="16khz")
        self.model = dac.DAC.load(model_path, n_codebooks=n_q)
        self.model.to(self.device).eval()

    def get_sampling_rate(self):
        return self.sample_rate

    def get_classification_embedding_size(self):
        return self.get_token_embedding_size()

    def get_token_embedding_size(self):
        # hardcoded for simplicity
        return 1024

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

        # DAC uses AudioSignal for preprocessing
        signal = AudioSignal(wav, sample_rate=self.sample_rate)
        signal.to(self.device)
        return signal

    def get_sequence_embeddings(self, audio: np.ndarray, **kwargs):
        signal = self._preprocess_audio(audio)

        with torch.no_grad():
            x = self.model.preprocess(signal.audio_data, signal.sample_rate)
            z, codes, latents, _, _ = self.model.encode(x)
            z = z.squeeze(0).transpose(0, 1)

        return z.cpu()

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        seq_embs = self.get_sequence_embeddings(audio)
        return seq_embs.mean(dim=0)  # (D,)

    def get_token_embeddings(self, audio: np.ndarray, **kwargs):
        return self.get_sequence_embeddings(audio)

    def get_embedding_layer(self):
        return self.embedding_dim

import os
import torch
import torchaudio
import numpy as np

from arch_eval import Model

import sys
# Need to git clone this repo: https://github.com/mubtasimahasan/FuseCodec
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../speech-token-modified')))
from codec.model_distill import Model as SpeechTokenizer

class SpeechTokenizerWrapper(Model):
    def __init__(self, config_path, ckpt_path, device="cuda", n_q=1, max_length=None):
        """
        :param config_path: path to config.yaml
        :param ckpt_path: path to checkpoint.ckpt
        :param n_q: number of quantizers to use
        :param max_length: max length in samples (optional)
        """
        super().__init__(model=None)
        self.device = torch.device(device)
        self.model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        self.model.eval().to(self.device)
        self.sample_rate = self.model.sample_rate
        self.n_q = n_q
        self.max_length = max_length

    def get_sampling_rate(self):
        return self.sample_rate

    def get_classification_embedding_size(self):
        return self.get_token_embedding_size()

    def get_token_embedding_size(self):
        # Use quantizer's defined vector dimension
        # codebook_dim = self.model.quantizer.dimension
        # print(f"[DEBUG] Quantizer Dimension: {codebook_dim}")
        # print(f"[DEBUG] Number of Quantizers (n_q): {self.n_q}")
        # print(f"[DEBUG] Total Token Embedding Size: {self.n_q * codebook_dim}")
        # print(f"[DEBUG] Self.model.quantizer: {self.model.quantizer}")
        # return self.n_q * codebook_dim
        return 768


    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            wav = audio.float()
        else:
            wav = torch.from_numpy(audio).float()

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # Add channel dimension
        elif wav.shape[0] > 1:
            wav = wav[:1, :]  # Convert to mono by taking first channel

        if self.sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, 16000, self.sample_rate)

        if self.max_length:
            wav = wav[:, :self.max_length]  # Truncate to max length

        return wav.unsqueeze(0).to(self.device)  # Final shape: (1, 1, T)

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        """
        Mean-pooled dense embedding across all quantizers and time.
        Output shape: (D,) where D = n_q * codebook_dim
        """
        seq_embs = self.get_sequence_embeddings(audio)
        pooled = seq_embs.mean(dim=0)  # (D,)
        return pooled.cpu()

    def get_sequence_embeddings(self, audio: np.ndarray, **kwargs):
        wav = self._preprocess_audio(audio)
        with torch.no_grad():
            # Run model.forward to get feature embeddings (B, T, D)
            _, _, features = self.model.forward(wav, n_q=self.n_q, layers=list(range(self.n_q)))
            # features shape: (B=1, T, D)
            features = features.squeeze(0)  # (T, D)
            return features.cpu()

    def get_token_embeddings(self, audio: np.ndarray, **kwargs):
        return self.get_sequence_embeddings(audio)

    def get_embedding_layer(self):
        return self.get_token_embedding_size()

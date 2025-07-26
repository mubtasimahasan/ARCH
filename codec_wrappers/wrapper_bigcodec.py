# wrapper_bigcodec.py

import torch
import torchaudio
import numpy as np
import os
from arch_eval import Model

import sys
# Need to git clone this repo: github.com/Aria-K-Alethia/BigCodec
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../BigCodec')))
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder import CodecDecoder


class BigCodecWrapper(Model):
    def __init__(self, ckpt_path, device="cuda", max_length=None):
        super().__init__(model=None)
        self.device = torch.device(device)
        self.sample_rate = 16000  # assumed
        self.max_length = max_length

        # Load BigCodec encoder and decoder
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.encoder = CodecEncoder().to(self.device).eval()
        self.decoder = CodecDecoder().to(self.device).eval()

        self.encoder.load_state_dict(ckpt['CodecEnc'])
        self.decoder.load_state_dict(ckpt['generator'])

        # You can infer codebook dimension like this after a forward pass if needed:
        self.embedding_dim = 1024  # Manually set for now — you can auto-infer it if needed

    def get_sampling_rate(self):
        return self.sample_rate

    def get_classification_embedding_size(self):
        return self.get_token_embedding_size()

    def get_token_embedding_size(self):
        return self.embedding_dim

    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            wav = audio.float()
        else:
            wav = torch.from_numpy(audio).float()

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.shape[0] > 1:
            wav = wav[:1, :]

        if self.sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, 16000, self.sample_rate)

        if self.max_length:
            wav = wav[:, :self.max_length]

        return wav.unsqueeze(0).to(self.device)  # (1, 1, T)

    def get_sequence_embeddings(self, audio: np.ndarray, **kwargs):
        wav = self._preprocess_audio(audio)  # (1, 1, T)
        with torch.no_grad():
            z = self.encoder(wav)  # (1, D, T)
            z = z.squeeze(0).transpose(0, 1)  # (T, D)
        return z.cpu()

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        seq = self.get_sequence_embeddings(audio)
        return seq.mean(dim=0).cpu()

    def get_token_embeddings(self, audio: np.ndarray, **kwargs):
        return self.get_sequence_embeddings(audio)

    def get_embedding_layer(self):
        return self.get_token_embedding_size()

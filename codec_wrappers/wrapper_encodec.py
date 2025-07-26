import os
import torch
import torchaudio
import numpy as np

from arch_eval import Model

from encodec import EncodecModel

class EncodecWrapper(Model):
    def __init__(self, device="cuda", n_q=8, max_length=None):
        """
        :param device: Device to run the model on ("cuda" or "cpu")
        :param n_q: Number of quantizers to use (supported: 2, 4, 8, 16, 32)
        :param max_length: Optional max length in samples
        """
        super().__init__(model=None)
        self.device = torch.device(device)
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth({2: 1.5, 4: 3.0, 8: 6.0, 16: 12.0, 32: 24.0}.get(n_q, 6.0))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.sample_rate = self.model.sample_rate
        self.n_q = n_q
        self.max_length = max_length

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
            wav = wav.unsqueeze(0)  # Add channel dimension
        elif wav.shape[0] > 1:
            wav = wav[:1, :]  # Convert to mono

        if self.sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, 16000, self.sample_rate)

        if self.max_length:
            wav = wav[:, :self.max_length]

        return wav.unsqueeze(0).to(self.device)  # Shape: (1, 1, T)

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        seq_embs = self.get_sequence_embeddings(audio)
        pooled = seq_embs.float().mean(dim=0)  # (D,)
        return pooled.cpu()

    def get_sequence_embeddings(self, audio: np.ndarray) -> torch.Tensor:
        """
        Return (T, D) embeddings from Encodec tokens by looking them up in the codebooks.
        """
        wav = self._preprocess_audio(audio)  # Shape: (1, 1, T)

        with torch.no_grad():
            encoded_frames = self.model.encode(wav)
            codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # (B, n_q, T)
            codes = codes[0].transpose(0, 1)  # (T, n_q)

            vq_layers = self.model.quantizer.vq.layers  # List of quantizers

            emb_list = []
            for i in range(codes.shape[1]):
                codebook = vq_layers[i].codebook  # (bins, dim)
                token_ids = codes[:, i]  # (T,)
                emb = codebook[token_ids]  # (T, dim)
                emb_list.append(emb)

            sequence_embeddings = torch.cat(emb_list, dim=-1)  # (T, n_q * dim)
            return sequence_embeddings.cpu()

    def get_token_embeddings(self, audio: np.ndarray, **kwargs):
        return self.get_sequence_embeddings(audio)

    def get_embedding_layer(self):
        return self.get_token_embedding_size()

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoFeatureExtractor
from configs.w2v2_wrapper import Wav2Vec2ModelWrapper
from wrapper import SpeechTokenizerWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Audio ===
wav = torch.randn(1, 16000 * 5)
wav_np = wav.squeeze(0).numpy()

# === Wav2Vec2 ===
wav2vec2_model = AutoModel.from_pretrained("facebook/wav2vec2-base").to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
w2v2 = Wav2Vec2ModelWrapper(wav2vec2_model, feature_extractor, device, max_length=80000)

print("== Wav2Vec2 ==")
print("SR:", w2v2.get_sampling_rate())
print("Class Emb Size:", w2v2.get_classification_embedding_size())
print("Token Emb Size:", w2v2.get_token_embedding_size())

pooled_w2v = w2v2.get_embeddings(wav_np)
print("Pooled Embedding Shape:", pooled_w2v.shape)

seq_w2v = w2v2.get_sequence_embeddings(wav_np)
print("Sequence Embedding Shape:", seq_w2v.shape)

tokens_w2v = w2v2.get_token_embeddings_old(wav_np)
print("Token Embeddings Shape:", tokens_w2v.shape)

# === SpeechTokenizer ===
st = SpeechTokenizerWrapper(
    config_path="../speech-token-modified/config.json",
    ckpt_path="../speech-token-modified/saved_files/fusecodec_distill/Model_best_dev.pt",
    device="cpu",
    n_q=8,  # Use 1 for semantic-only tokens
    max_length=80000
)

print("\n== SpeechTokenizer ==")
print("SR:", st.get_sampling_rate())
print("Class Emb Size:", st.get_classification_embedding_size())
print("Token Emb Size:", st.get_token_embedding_size())

pooled_st = st.get_embeddings(wav_np)
print("Pooled Embedding Shape:", pooled_st.shape)

seq_st = st.get_sequence_embeddings(wav_np)
print("Sequence Embedding Shape:", seq_st.shape)

tokens_st = st.get_token_embeddings(wav_np)
print("Token Embeddings Shape:", tokens_st.shape)

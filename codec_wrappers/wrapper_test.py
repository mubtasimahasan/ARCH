import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoFeatureExtractor
from configs.w2v2_wrapper import Wav2Vec2ModelWrapper
from codec_wrappers.wrapper_speechtokenizer import SpeechTokenizerWrapper
from codec_wrappers.wrapper_encodec import EncodecWrapper
from ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
from codec_wrappers.wrapper_facodec import FACodecWrapper
from codec_wrappers.wrapper_dac import DACWrapper
from codec_wrappers.wrapper_bigcodec import BigCodecWrapper


device = "cuda" if torch.cuda.is_available() else "cpu"

wav = torch.randn(1, 16000 * 3)
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
    n_q=8,
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


# === Encodec ===
enc = EncodecWrapper(
    device=device,
    n_q=8,  # Use 2, 4, 8, 16, or 32 depending on bandwidth
    max_length=80000
)

print("\n== Encodec ==")
print("SR:", enc.get_sampling_rate())
print("Class Emb Size:", enc.get_classification_embedding_size())
print("Token Emb Size:", enc.get_token_embedding_size())

pooled_enc = enc.get_embeddings(wav_np)
print("Pooled Embedding Shape:", pooled_enc.shape)

seq_enc = enc.get_sequence_embeddings(wav_np)
print("Sequence Embedding Shape:", seq_enc.shape)

tokens_enc = enc.get_token_embeddings(wav_np)
print("Token Embeddings Shape:", tokens_enc.shape)


# === FACodec ===
fa_encoder = FACodecEncoder(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
)

fa_decoder = FACodecDecoder(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)

# Load pretrained weights
encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

fa_encoder.load_state_dict(torch.load(encoder_ckpt))
fa_decoder.load_state_dict(torch.load(decoder_ckpt))

fa_codec = FACodecWrapper(fa_encoder, fa_decoder, device=device, max_length=80000)

print("\n== FACodec ==")
print("SR:", fa_codec.get_sampling_rate())
print("Class Emb Size:", fa_codec.get_classification_embedding_size())
print("Token Emb Size:", fa_codec.get_token_embedding_size())

pooled_fa = fa_codec.get_embeddings(wav_np)
print("Pooled Embedding Shape:", pooled_fa.shape)

seq_fa = fa_codec.get_sequence_embeddings(wav_np)
print("Sequence Embedding Shape:", seq_fa.shape)

tokens_fa = fa_codec.get_token_embeddings(wav_np)
print("Token Embeddings Shape:", tokens_fa.shape)


# === DAC ===
print("\n== DAC ==")
dac = DACWrapper(device=device, n_q=8, max_length=80000)

print("SR:", dac.get_sampling_rate())
print("Class Emb Size:", dac.get_classification_embedding_size())
print("Token Emb Size:", dac.get_token_embedding_size())

pooled_dac = dac.get_embeddings(wav_np)
print("Pooled Embedding Shape:", pooled_dac.shape)

seq_dac = dac.get_sequence_embeddings(wav_np)
print("Sequence Embedding Shape:", seq_dac.shape)

tokens_dac = dac.get_token_embeddings(wav_np)
print("Token Embeddings Shape:", tokens_dac.shape)


# === FACodec ===
bigcodec = BigCodecWrapper(
    ckpt_path="/teamspace/studios/this_studio/BigCodec/bigcodec.pt",
    device=device,
    max_length=80000
)

print("\n== BigCodec ==")
print("SR:", bigcodec.get_sampling_rate())
print("Class Emb Size:", bigcodec.get_classification_embedding_size())
print("Token Emb Size:", bigcodec.get_token_embedding_size())

pooled_big = bigcodec.get_embeddings(wav_np)
print("Pooled Embedding Shape:", pooled_big.shape)

seq_big = bigcodec.get_sequence_embeddings(wav_np)
print("Sequence Embedding Shape:", seq_big.shape)

tokens_big = bigcodec.get_token_embeddings(wav_np)
print("Token Embeddings Shape:", tokens_big.shape)
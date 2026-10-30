import torch
import laion_clap
import numpy as np

CKPT = 'weights/music_audioset_epoch_15_esc_90.14.pt'

def load_clap(checkpoint=CKPT):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model.load_ckpt(ckpt=str(checkpoint), verbose=False)
    model = model.to(device).eval()
    return model

def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def embed_text(model, texts, batch_size=16):
    chunks = []
    for i in range(0, len(texts), batch_size):
        with torch.no_grad():
            e = model.get_text_embedding(texts[i:i+batch_size], use_tensor=True)
        chunks.append(e.detach().float().cpu().numpy())
    return normalize(np.concatenate(chunks))

def embed_audio(model, paths, batch_size=8):
    chunks = []
    for i in range(0, len(paths), batch_size):
        with torch.no_grad():
            e = model.get_audio_embedding_from_filelist(x=[str(p) for p in paths[i:i+batch_size]], use_tensor=True)
        chunks.append(e.detach().float().cpu().numpy())
    return normalize(np.concatenate(chunks))

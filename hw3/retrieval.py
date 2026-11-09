import json
import torch
import laion_clap
import numpy as np

from pathlib import Path

from common import read_captions, audio_files

CAPTIONS_PATH = Path("test.csv")
AUDIO_DIR = Path("audio_mp3")
OUTPUT_PATH = Path("outputs/retrieval.json")
CACHE_PATH = Path("outputs/musiccaps_embeddings.npz")
CKPT_PATH = Path("av-benchmark/weights/music_audioset_epoch_15_esc_90.14.pt")

BATCH_SIZE = 8


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True), 1e-12
    )


def load_clap():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = laion_clap.CLAP_Module(
        enable_fusion=False,
        amodel="HTSAT-base",
        device=device,
    )
    model.load_ckpt(str(CKPT))
    model.eval()

    print(f"CLAP loaded on {device}")
    return model


def embed_text(model, captions):
    embeddings = []

    with torch.inference_mode():
        for start in range(0, len(captions), BATCH_SIZE):
            batch = captions[start:start + BATCH_SIZE]
            emb = model.get_text_embedding(batch, use_tensor=False)
            embeddings.append(np.asarray(emb))

    return np.concatenate(embeddings, axis=0)


def embed_audio(model, files):
    embeddings = []

    with torch.inference_mode():
        for start in range(0, len(files), BATCH_SIZE):
            batch = files[start:start + BATCH_SIZE]

            emb = model.get_audio_embedding_from_filelist(
                x=[str(f) for f in batch],
                use_tensor=False,
            )
            embeddings.append(np.asarray(emb))

            print(
                f"Audio embedded: "
                f"{min(start + BATCH_SIZE, len(files))}/{len(files)}"
            )

    return np.concatenate(embeddings, axis=0)


def main():
    rows = read_captions(CAPTIONS_PATH)
    files = audio_files(AUDIO_DIR)

    if not rows:
        raise ValueError("No captions found")
    if not files:
        raise ValueError("No audio files found")
    if not CKPT.exists():
        raise FileNotFoundError(f"CLAP checkpoint not found: {CKPT}")

    names = [f.stem for f in files]
    paths = [str(f.resolve()) for f in files]
    checkpoint = str(CKPT.resolve())

    model = load_clap()
    audio_emb = None

    if CACHE_PATH.exists():
        with np.load(CACHE_PATH, allow_pickle=False) as z:
            if (
                list(z["paths"]) == paths
                and str(z["checkpoint"]) == checkpoint
            ):
                audio_emb = z["embeddings"]
                print("Reusing cached MusicCaps embeddings")

    if audio_emb is None:
        audio_emb = embed_audio(model, files)

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE_PATH,
            paths=np.array(paths),
            embeddings=audio_emb,
            checkpoint=checkpoint,
        )

    captions = [caption for _, caption in rows]
    text_emb = embed_text(model, captions)

    audio_emb = normalize(audio_emb)
    text_emb = normalize(text_emb)

    similarities = text_emb @ audio_emb.T
    best = similarities.argmax(axis=1)

    result = {
        str(sample_id): names[int(idx)]
        for (sample_id, _), idx in zip(rows, best)
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {len(result)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

import json
import numpy as np

from pathlib import Path

from common import read_captions, audio_files
from clap_utils import load_clap, embed_text, embed_audio, CKPT

CAPTIONS_PATH = Path("test.csv")
AUDIO_DIR = Path("audio_mp3")
OUTPUT_PATH = Path("outputs/retrieval.json")
CACHE_PATH = Path("outputs/musiccaps_embeddings.npz")

BATCH_SIZE = 8

def main():
    rows = read_captions(CAPTIONS_PATH)
    files = audio_files(AUDIO_DIR)

    names = [f.stem for f in files]

    model = load_clap(CKPT)

    if CACHE_PATH.exists():
        z = np.load(CACHE_PATH, allow_pickle=False)
        if list(z['paths']) == [str(f.resolve()) for f in files] and str(z['checkpoint']) == str(Path(CKPT).resolve()):
            audio_emb = z['embeddings']; print('Reusing cached MusicCaps embeddings')

    else:
        audio_emb = embed_audio(model, files, BATCH_SIZE)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_PATH, paths=np.array([str(f.resolve()) for f in files]), embeddings=audio_emb, checkpoint=str(Path(CKPT).resolve()))

    text_emb = embed_text(model, [caption for _, caption in rows])
    best = (text_emb @ audio_emb.T).argmax(axis=1)
    result = {sample_id: names[int(idx)] for (sample_id, _), idx in zip(rows, best)}

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f'Saved {len(result)} results to {OUTPUT_PATH}; audio candidates: {len(files)}')

if __name__ == '__main__': main()

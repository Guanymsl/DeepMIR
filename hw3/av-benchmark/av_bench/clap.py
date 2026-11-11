"""LAION CLAP scoring with the benchmark audio preprocessing."""
import csv
from pathlib import Path

import laion_clap
import torch
from torch.utils.data import DataLoader

from av_bench.data.audio_dataset import AudioDataset


@torch.inference_mode()
def evaluate_clap(pred: Path, captions_path: Path, audio_length: float, batch_size: int):
    checkpoint = Path(__file__).resolve().parent.parent / 'weights/music_audioset_epoch_15_esc_90.14.pt'

    with captions_path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    clap_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = laion_clap.CLAP_Module(
        enable_fusion=False,
        amodel='HTSAT-base',
        device=clap_device,
    )
    model.load_ckpt(str(checkpoint), verbose=False)
    model.eval()

    caption_map = {str(row['name']): row['caption'] for row in rows}
    if not caption_map:
        raise ValueError(f'No captions found in {captions_path}')

    audio_paths = {}
    for path in pred.iterdir():
        if path.is_file() and path.suffix.lower() in {'.wav', '.flac'}:
            if path.stem in audio_paths:
                raise ValueError(f'Multiple audio files for sample {path.stem}')
            audio_paths[path.stem] = path
    missing = caption_map.keys() - audio_paths.keys()
    if missing:
        raise ValueError(f'Missing generated audio for: {sorted(missing)}')

    dataset = AudioDataset(
        [audio_paths[sid] for sid in caption_map],
        audio_length=audio_length,
        sr=48_000,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    scores = {}

    with torch.inference_mode():
        for wav, filenames in loader:
            texts = [caption_map[sid] for sid in filenames]
            text_emb = model.get_text_embedding(texts, use_tensor=True)
            audio_emb = model.get_audio_embedding_from_data(
                wav.squeeze(1).to(clap_device),
                use_tensor=True,
            )
            values = torch.cosine_similarity(text_emb, audio_emb, dim=-1)
            for sid, score in zip(filenames, values):
                scores[sid] = float(score.item())

    return sum(scores.values()) / len(scores), scores

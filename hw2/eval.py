import json
import random
import itertools
import numpy as np
import pandas as pd

from pathlib import Path
from miditok import REMI
from tqdm import tqdm

from MusDr.musdr.side_utils import (
    get_bars_crop,
    get_pitch_histogram,
    compute_histogram_entropy,
    get_onset_xor_distance,
)

TOKENIZER_PATH = Path('outputs/tokens/tokenizer.json')
ORIGINAL_PATH = Path("outputs/tokens")
GENERATED_DIR = Path('outputs/generated/A')
OUTPUT_DIR = Path('outputs/evaluation')

def token_ids(tokenizer):
    vocab = tokenizer.vocab

    bar_id = vocab['Bar_None']
    pos_keys = sorted(
        (key for key in vocab if key.startswith('Position_')),
        key=lambda key: int(key.split('_', 1)[1]),
    )
    pitch_keys = sorted(
        (key for key in vocab if key.startswith('Pitch_')),
        key=lambda key: int(key.split('_', 1)[1]),
    )

    return bar_id, [vocab[key] for key in pos_keys], [vocab[key] for key in pitch_keys]

def normalized_sequence(ids, bar_id):
    if ids and ids[-1] == bar_id:
        return ids[:-1]
    return ids

def pitch_entropy(ids, window_size, bar_id, pitch_ids):
    ids = normalized_sequence(ids, bar_id)

    n_bars = ids.count(bar_id)
    window_size = min(window_size, n_bars)
    if window_size == 0:
        return np.nan

    values = []
    for start in range(n_bars - window_size + 1):
        segment = get_bars_crop(ids, start, start + window_size - 1, bar_id)
        hist = get_pitch_histogram(segment, pitch_evs=pitch_ids)
        if hist is None:
            continue

        value = compute_histogram_entropy(hist)
        if np.isfinite(value):
            values.append(float(value))

    return float(np.mean(values)) if values else np.nan

def groove_similarity(ids, bar_id, pos_ids, pitch_ids, max_pairs=1000):
    ids = normalized_sequence(ids, bar_id)

    n_bars = ids.count(bar_id)
    if n_bars < 2:
        return np.nan

    bar_seqs = [get_bars_crop(ids, i, i, bar_id) for i in range(n_bars)]
    pairs = list(itertools.combinations(range(n_bars), 2))
    if len(pairs) > max_pairs:
        pairs = random.Random(42).sample(pairs, max_pairs)

    values = []
    for i, j in pairs:
        distance = get_onset_xor_distance(
            bar_seqs[i], bar_seqs[j], bar_id, pos_ids, pitch_evs=pitch_ids
        )
        if np.isfinite(distance):
            values.append(float(1.0 - distance))

    return float(np.mean(values)) if values else np.nan

def main():
    tokenizer = REMI(params=str(TOKENIZER_PATH))
    bar_id, pos_ids, pitch_ids = token_ids(tokenizer)
    paths = sorted(GENERATED_DIR.glob('*/*/*.json'))

    rows = []
    for path in tqdm(paths, desc='Evaluating Task A'):
        with path.open(encoding='utf-8') as f:
            ids = json.load(f)

        rows.append({
            'config': path.parent.parent.name,
            'song_name': path.stem,
            'file': str(path),
            'bars': normalized_sequence(ids, bar_id).count(bar_id),
            'H1': pitch_entropy(ids, 1, bar_id, pitch_ids),
            'H4': pitch_entropy(ids, 4, bar_id, pitch_ids),
            'GS': groove_similarity(ids, bar_id, pos_ids, pitch_ids),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'metrics.csv', index=False)

    summary = (
        df.groupby('config', as_index=False)
        .agg(samples=('song_name', 'count'), mean_H1=('H1', 'mean'),
             mean_H4=('H4', 'mean'), mean_GS=('GS', 'mean'))
        .sort_values('config')
    )
    summary.to_csv(OUTPUT_DIR/ 'summary.csv', index=False)

    print('\nTask A Evaluation Summary:')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\nSaved: {OUTPUT_DIR / "metrics.csv"}')
    print(f'Saved: {OUTPUT_DIR / "summary.csv"}')

if __name__ == '__main__':
    main()

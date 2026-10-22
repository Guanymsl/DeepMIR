import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from MusDr.mueller_audio_tools.scapeplot import (
    compute_fitness_scape_plot,
    normalization_properties_SSM,
)
from MusDr.mueller_audio_tools.ssm_features import (
    compute_SM_from_filename,
    compute_tempo_rel_set,
)

AUDIO_DIR = Path('outputs/generated/A')
SCAPE_DIR = Path('outputs/evaluation/scplot')
OUTPUT_DIR = Path('outputs/evaluation')

TEMPO_REL_SET = compute_tempo_rel_set(0.5, 2, 7)
SHIFT_SET = np.arange(12)

def compute_si(fitness, low_sec, high_sec=None, sample_rate=2):
    start = int(low_sec * sample_rate) - 1
    end = int(high_sec * sample_rate) if high_sec is not None else fitness.shape[0]
    if start >= fitness.shape[0]:
        return 0.0
    window = fitness[max(start, 0):end]
    return float(np.max(window)) if window.size else 0.0

def get_fitness(wav_path, cache_path):
    if cache_path.exists():
        return np.load(cache_path)

    _, _, _, _, ssm, _ = compute_SM_from_filename(
        str(wav_path),
        tempo_rel_set=TEMPO_REL_SET,
        shift_set=SHIFT_SET,
        thresh=0.25,
        penalty=-2,
    )

    ssm = normalization_properties_SSM(ssm)
    fitness = compute_fitness_scape_plot(ssm)[0]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, fitness)

    return fitness

def main():
    wavs = sorted(AUDIO_DIR.glob('*/*.wav'))

    rows = []
    for path in tqdm(wavs, desc='Computing MusDr SI'):
        config = path.parent.name
        cache = SCAPE_DIR / config / (path.stem + '_fitness.npy')
        fitness = get_fitness(path, cache)
        rows.append({
            'config': config,
            'song_name': path.stem,
            'file': str(path),
            'SI_short': compute_si(fitness, 3, 8),
            'SI_mid': compute_si(fitness, 8, 15),
            'SI_long': compute_si(fitness, 15),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / 'si_metrics.csv', index=False)

    summary = df.groupby('config', as_index=False).agg(
        samples=('song_name', 'count'),
        mean_SI_short=('SI_short', 'mean'),
        mean_SI_mid=('SI_mid', 'mean'),
        mean_SI_long=('SI_long', 'mean'),
    ).sort_values('config')

    summary.to_csv(OUTPUT_DIR / 'si_summary.csv', index=False)

    print('\nMusDr SI Summary:')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\nSaved: {OUTPUT_DIR / "si_metrics.csv"}')
    print(f'Saved: {OUTPUT_DIR / "si_summary.csv"}')

if __name__ == '__main__':
    main()

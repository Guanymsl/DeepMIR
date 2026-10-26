import json

from pathlib import Path
from miditok import REMI
from tqdm import tqdm
from symusic import Score, Track

from tokenization import corrected_score

DATA_DIR = Path('data/POP909/training')
TOKENIZER_PATH = Path('outputs/tokens/tokenizer.json')
OUTPUT_PATH = Path('outputs/bonus/tokens_bonus.json')

def select_track(score, name):
    tracks = [track for track in score.tracks if track.name.strip().upper() == name]
    if not tracks:
        return None

    result = Score(score.ticks_per_quarter)
    result.tempos = list(score.tempos)
    result.time_signatures = list(score.time_signatures)

    merged = Track(program=0, is_drum=False, name=name)
    for track in tracks:
        merged.notes.extend(track.notes)
    merged.notes.sort(key=lambda n: (n.time, n.pitch, n.duration))

    result.tracks = [merged]
    return result

def encode_track(tokenizer, score, name):
    selected = select_track(score, name)
    if selected is None:
        return None

    encoded = tokenizer.encode(selected)
    if isinstance(encoded, list):
        encoded = encoded[0]
    return list(encoded.ids)

def main():
    tokenizer = REMI(params=str(TOKENIZER_PATH))
    paths = sorted(DATA_DIR.rglob('*.mid'))

    results = []
    for path in tqdm(paths, desc='Tokenizing Bonus'):
        beat_path = path.parent / 'beat_midi.txt'

        score, _ = corrected_score(path, beat_path)

        melody = encode_track(tokenizer, score, 'MELODY')
        piano = encode_track(tokenizer, score, 'PIANO')
        if melody is None or piano is None:
            continue

        results.append({
            'song_name': path.stem,
            'source_midi': str(path),
            'melody_ids': melody,
            'piano_ids': piano,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(results, f)
    print(f'Saved {len(results)} songs')

if __name__ == '__main__':
    main()

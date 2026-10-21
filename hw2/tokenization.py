import json
import pretty_midi

from pathlib import Path
from miditok import REMI, TokenizerConfig
from tqdm import tqdm
from symusic import Score, Track, TimeSignature

def read_beats(beat_path):
    times, downbeats = [], []
    with Path(beat_path).open() as f:
        for line in f:
            parts = line.split()
            times.append(float(parts[0]))
            if float(parts[2]) == 1.0:
                downbeats.append(len(times) - 1)
    return times, downbeats

def beat_meter(beat_path):
    times, downbeats = read_beats(beat_path)
    return [(times[a], b - a) for a, b in zip(downbeats, downbeats[1:])]

def corrected_score(midi_path, beat_path):
    times, downbeats = read_beats(beat_path)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    score = Score(str(midi_path))

    meters = [(a, b - a) for a, b in zip(downbeats, downbeats[1:])]
    changes = [(0, meters[0][1], 4)]
    previous = meters[0][1]

    for index, numerator in meters:
        tick = midi.time_to_tick(times[index])
        if numerator != previous:
            changes.append((tick, numerator, 4))
            previous = numerator

    score.time_signatures = [TimeSignature(t, n, d) for t, n, d in changes]
    return score, changes

def create(meters):
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 16},
        num_velocities=32,
        use_tempos=True,
        use_time_signatures=True,
        time_signature_range={4: sorted(meters)},
        use_programs=False,
        use_chords=False,
    )
    return REMI(config)

def normalize(encoded):
    return encoded if isinstance(encoded, list) else [encoded]

def main():
    dataset = Path('data/POP909/training')
    dest = Path('outputs/tokens')
    files = sorted((d.name, d / f'{d.name}.mid', d / 'beat_midi.txt') for d in dataset.iterdir() if d.is_dir())

    meters = {n for _, _, beat in files for _, n in beat_meter(beat)}
    tokenizer = create(meters)
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer.save(dest / 'tokenizer.json')

    results = []
    for name, midi, beat in tqdm(files, desc=f'Tokenizing', unit='song'):
        score, _ = corrected_score(midi, beat)

        merged = Track(program=0, is_drum=False, name='PIANO_MERGED')
        for track in score.tracks:
            if not track.is_drum:
                merged.notes.extend(track.notes)

        merged.notes.sort(key=lambda n: (n.time, n.pitch, n.duration))
        score.tracks = [merged]

        sequences = normalize(tokenizer.encode(score))
        results.append({
            'song_name': name,
            'source_midi': str(midi),
            'sequences': [{'ids': seq.ids, 'tokens': seq.tokens} for seq in sequences],
        })

    with (dest / 'tokens.json').open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    print(f'Saved {len(results)} songs')

if __name__ == '__main__':
    main()

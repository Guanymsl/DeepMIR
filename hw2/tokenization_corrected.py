import argparse
import json
import tempfile
from pathlib import Path

import pretty_midi
from miditok import CPWord, REMI, TokenizerConfig
from symusic import Score, Track
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, choices=['remi', 'cpword'])
    return parser.parse_args()


def beat_meter(beat_path):
    beat_times, downbeats = [], []
    with beat_path.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            beat_times.append(float(parts[0]))
            if float(parts[2]) == 1.0:
                downbeats.append(len(beat_times) - 1)
    if len(downbeats) < 2:
        raise ValueError(f'Cannot infer meter from {beat_path}: fewer than 2 downbeats')
    # The assistant sample interprets successive beat rows as quarter-note beats.
    meters = [(beat_times[a], b - a) for a, b in zip(downbeats, downbeats[1:])]
    if any(n < 1 or n > 16 for _, n in meters):
        raise ValueError(f'Unexpected beats per bar in {beat_path}: {meters[:8]}')
    return meters


def corrected_score(midi_path, beat_path):
    """Correct time signatures only; keep notes, note times, velocities and tempos."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    meters = beat_meter(beat_path)
    # The first detected meter also applies before the first annotated downbeat.
    changes = [pretty_midi.TimeSignature(meters[0][1], 4, 0.0)]
    previous = meters[0][1]
    for time_sec, beats in meters[1:]:
        if beats != previous:
            changes.append(pretty_midi.TimeSignature(beats, 4, time_sec))
            previous = beats
    midi.time_signature_changes = changes
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'corrected.mid'
        midi.write(str(path))
        return Score(str(path)), [(ts.numerator, ts.denominator, ts.time) for ts in changes]


def create(kind, meters):
    # Include all meters discovered in the training dataset in the MidiTok vocabulary.
    signature_range = {4: sorted(meters)}
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 16},
        num_velocities=32,
        use_tempos=True,
        use_time_signatures=True,
        time_signature_range=signature_range,
        use_programs=False,
        use_chords=False,
    )
    return (REMI if kind == 'remi' else CPWord)(config)


def normalize(encoded):
    return encoded if isinstance(encoded, list) else [encoded]


def main():
    args = parse_args()
    dataset = Path('data/POP909/training')
    dest = Path('outputs/tokens') / args.type
    files = sorted((d.name, d / f'{d.name}.mid', d / 'beat_midi.txt')
                   for d in dataset.iterdir() if d.is_dir())
    for _, midi, beat in files:
        if not midi.is_file() or not beat.is_file():
            raise FileNotFoundError(f'Missing {midi} or {beat}')

    # Build vocabulary from all time signatures before encoding.
    meters = {n for _, _, beat in files for _, n in beat_meter(beat)}
    print('Detected beats per bar:', sorted(meters))
    tokenizer = create(args.type, meters)
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer.save(dest / 'tokenizer.json')

    results = []
    for name, midi, beat in tqdm(files, desc=f'Tokenizing {args.type.upper()}', unit='song'):
        score, changes = corrected_score(midi, beat)
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

import argparse
import json
import tempfile
import pretty_midi
import mido

from pathlib import Path
from bisect import bisect_right
from miditok import CPWord, REMI, TokenizerConfig
from symusic import Score, Track
from tqdm import tqdm

PPQ = 480

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', required=True, choices=['remi', 'cpword'])
    return parser.parse_args()

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

def second_to_tick(t, times):
    i = max(0, min(bisect_right(times, t) - 1, len(times) - 2))
    return max(0, round((i + (t - times[i]) / (times[i + 1] - times[i])) * PPQ))

def corrected_score(midi_path, beat_path):
    times, downbeats = read_beats(beat_path)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    out = mido.MidiFile(ticks_per_beat=PPQ)

    meta = mido.MidiTrack()
    out.tracks.append(meta)
    events = []

    for i, (start, end) in enumerate(zip(times, times[1:])):
        tempo = round((end - start) * 1_000_000)
        events.append((i * PPQ, 0, mido.MetaMessage('set_tempo', tempo=tempo)))

    meters = [(a, b - a) for a, b in zip(downbeats, downbeats[1:])]
    last_meter = None
    for beat_index, numerator in meters:
        if numerator != last_meter:
            events.append((beat_index * PPQ, 1, mido.MetaMessage(
                'time_signature', numerator=numerator, denominator=4
            )))
            last_meter = numerator

    def add_events(track, entries):
        previous_tick = 0
        for tick, _, message in sorted(entries, key=lambda x: (x[0], x[1])):
            track.append(message.copy(time=tick - previous_tick))
            previous_tick = tick
        track.append(mido.MetaMessage('end_of_track', time=0))

    add_events(meta, events)

    for channel_index, instrument in enumerate(midi.instruments):
        track = mido.MidiTrack()
        out.tracks.append(track)

        channel = 9 if instrument.is_drum else channel_index % 15
        if not instrument.is_drum and channel >= 9:
            channel += 1
        instrument_events = [(0, 0, mido.Message(
            'program_change', channel=channel, program=instrument.program
        ))]
        for note in instrument.notes:
            start = second_to_tick(note.start, times)
            end = max(start + 1, second_to_tick(note.end, times))
            instrument_events.append((start, 2, mido.Message(
                'note_on', channel=channel, note=note.pitch, velocity=note.velocity
            )))
            instrument_events.append((end, 1, mido.Message(
                'note_off', channel=channel, note=note.pitch, velocity=0
            )))
        for cc in instrument.control_changes:
            instrument_events.append((second_to_tick(cc.time, times), 1, mido.Message(
                'control_change', channel=channel, control=cc.number, value=cc.value
            )))
        for bend in instrument.pitch_bends:
            instrument_events.append((second_to_tick(bend.time, times), 1, mido.Message(
                'pitchwheel', channel=channel, pitch=bend.pitch
            )))
        add_events(track, instrument_events)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'corrected.mid'
        out.save(path)
        score = Score(str(path))

    changes = []
    previous_meter = None
    for index, numerator in meters:
        if numerator != previous_meter:
            changes.append((numerator, 4, times[index]))
            previous_meter = numerator
    return score, changes

def create(kind, meters):
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
    return (REMI if kind == 'remi' else CPWord)(config)

def normalize(encoded):
    return encoded if isinstance(encoded, list) else [encoded]

def main():
    args = parse_args()

    dataset = Path('data/POP909/training')
    dest = Path('outputs/tokens') / args.type
    files = sorted((d.name, d / f'{d.name}.mid', d / 'beat_midi.txt') for d in dataset.iterdir() if d.is_dir())

    meters = {n for _, _, beat in files for _, n in beat_meter(beat)}
    tokenizer = create(args.type, meters)
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer.save(dest / 'tokenizer.json')

    results = []
    for name, midi, beat in tqdm(files, desc=f'Tokenizing {args.type.upper()}', unit='song'):
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

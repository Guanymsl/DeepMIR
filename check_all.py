import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pretty_midi
from miditok import REMI
from symusic import Track
from tqdm import tqdm

from tokenization_corrected import corrected_score, read_beats, beat_offset, PPQ

DATA_DIR = Path('data/POP909/training')
TOKENIZER_PATH = Path('outputs/tokens/remi/tokenizer.json')
OUTPUT_DIR = Path('outputs')
MAX_TIME_ERROR_MS = 2.0
MAX_BEAT_ERROR_MS = 2.0
MAX_BAR_DIFFERENCE = 1

FIELDS = [
    'song', 'status', 'notes_original', 'notes_corrected',
    'pitch_velocity_match', 'max_start_error_ms', 'max_end_error_ms',
    'first_beat_s', 'first_downbeat_index', 'expected_bars',
    'token_bars', 'bar_difference', 'tokens', 'unknown_tokens',
    'meters', 'tempo_events', 'max_beat_error_ms', 'issues',
]


def get_original_notes(midi):
    return sorted(
        (n.pitch, n.velocity, n.start, n.end)
        for inst in midi.instruments if not inst.is_drum
        for n in inst.notes
    )


def get_corrected_notes(score):
    seconds = score.to('second', min_dur=0.000001)
    return sorted(
        (n.pitch, n.velocity, n.time, n.time + n.duration)
        for track in seconds.tracks if not track.is_drum
        for n in track.notes
    )


def compare_notes(original, score, result, issues):
    a = get_original_notes(original)
    b = get_corrected_notes(score)
    result['notes_original'] = len(a)
    result['notes_corrected'] = len(b)
    if len(a) != len(b):
        issues.append('NOTE_COUNT_MISMATCH')
        return
    a.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    b.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    match = all(x[:2] == y[:2] for x, y in zip(a, b))
    result['pitch_velocity_match'] = match
    if not match:
        issues.append('PITCH_VELOCITY_MISMATCH')
        return
    if not a:
        issues.append('NO_NOTES')
        return
    starts = np.array([abs(x[2] - y[2]) * 1000 for x, y in zip(a, b)])
    ends = np.array([abs(x[3] - y[3]) * 1000 for x, y in zip(a, b)])
    result['max_start_error_ms'] = float(starts.max())
    result['max_end_error_ms'] = float(ends.max())
    if starts.max() > MAX_TIME_ERROR_MS:
        issues.append('START_TIME_ERROR')
    if ends.max() > MAX_TIME_ERROR_MS:
        issues.append('END_TIME_ERROR')


def tick_to_seconds(tick, score):
    tpq = score.ticks_per_quarter
    events = sorted(score.tempos, key=lambda e: e.time)
    last_tick, seconds, bpm = 0, 0.0, 120.0
    for event in events:
        if event.time > tick:
            break
        seconds += (event.time - last_tick) * 60.0 / (tpq * bpm)
        last_tick, bpm = event.time, event.qpm
    return seconds + (tick - last_tick) * 60.0 / (tpq * bpm)


def compare_beats(score, times, result, issues):
    result['tempo_events'] = len(score.tempos)
    if not score.tempos:
        issues.append('NO_TEMPO')
        return
    offset = beat_offset(times)
    errors = [
        abs(tick_to_seconds(offset + i * PPQ, score) - t) * 1000
        for i, t in enumerate(times)
    ]
    result['max_beat_error_ms'] = float(max(errors))
    if max(errors) > MAX_BEAT_ERROR_MS:
        issues.append('BEAT_TIME_ERROR')


def compare_meters(score, downbeats, result, issues):
    meters = [(a, b - a) for a, b in zip(downbeats, downbeats[1:])]
    result['meters'] = ','.join(map(str, sorted({n for _, n in meters})))
    if not meters:
        issues.append('NO_COMPLETE_BARS')
        return
    signatures = sorted(score.time_signatures, key=lambda e: e.time)
    if not signatures:
        issues.append('NO_TIME_SIGNATURE')
        return
    offset = result['_offset']
    for beat_index, numerator in meters:
        if not 1 <= numerator <= 16:
            issues.append('INVALID_METER')
            return
        tick = offset + beat_index * PPQ
        active = None
        for ts in signatures:
            if ts.time <= tick:
                active = ts
            else:
                break
        if active is None or (active.numerator, active.denominator) != (numerator, 4):
            issues.append('TIME_SIGNATURE_MISMATCH')
            return


def tokenize(score, tokenizer, result, issues):
    merged = Track(program=0, is_drum=False, name='PIANO_MERGED')
    for track in score.tracks:
        if not track.is_drum:
            merged.notes.extend(track.notes)
    merged.notes.sort(key=lambda n: (n.time, n.pitch, n.duration))
    score.tracks = [merged]
    encoded = tokenizer.encode(score)
    sequences = encoded if isinstance(encoded, list) else [encoded]
    tokens = [token for seq in sequences for token in seq.tokens]
    result['tokens'] = len(tokens)
    result['token_bars'] = tokens.count('Bar_None')
    result['unknown_tokens'] = sum('UNK' in token.upper() for token in tokens)
    result['bar_difference'] = result['token_bars'] - result['expected_bars']
    if abs(result['bar_difference']) > MAX_BAR_DIFFERENCE:
        issues.append('BAR_COUNT_MISMATCH')
    if result['unknown_tokens']:
        issues.append('UNKNOWN_TOKENS')
    if not tokens:
        issues.append('EMPTY_TOKENS')


def check_song(folder, tokenizer):
    name = folder.name
    midi_path = folder / f'{name}.mid'
    beat_path = folder / 'beat_midi.txt'
    result = {field: None for field in FIELDS}
    result['song'] = name
    issues = []
    if not midi_path.is_file() or not beat_path.is_file():
        raise FileNotFoundError(f'Missing MIDI or beats in {folder}')
    times, downbeats = read_beats(beat_path)
    if len(times) < 2 or len(downbeats) < 2:
        raise ValueError('Insufficient beats/downbeats')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Non-increasing beat timestamps')
    if any(i < 0 or i >= len(times) for i in downbeats):
        raise ValueError('Invalid downbeat index')
    result['first_beat_s'] = times[0]
    result['first_downbeat_index'] = downbeats[0]
    result['expected_bars'] = len(downbeats) - 1
    result['_offset'] = beat_offset(times)
    original = pretty_midi.PrettyMIDI(str(midi_path))
    score, _ = corrected_score(midi_path, beat_path)
    compare_notes(original, score, result, issues)
    compare_beats(score, times, result, issues)
    compare_meters(score, downbeats, result, issues)
    tokenize(score, tokenizer, result, issues)
    result['status'] = 'WARNING' if issues else 'PASS'
    result['issues'] = ';'.join(issues)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = REMI(params=TOKENIZER_PATH)
    folders = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    results = []
    for folder in tqdm(folders, desc='Checking POP909'):
        try:
            result = check_song(folder, tokenizer)
        except Exception as exc:
            result = {field: None for field in FIELDS}
            result.update(song=folder.name, status='ERROR', issues=f'{type(exc).__name__}: {exc}')
        results.append(result)
    csv_path = OUTPUT_DIR / 'check_dataset.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    statuses = Counter(r['status'] for r in results)
    issues = Counter()
    for result in results:
        if result['status'] == 'WARNING':
            issues.update(result['issues'].split(';'))
    summary = {
        'total': len(results), 'pass': statuses['PASS'],
        'warning': statuses['WARNING'], 'error': statuses['ERROR'],
        'issues': dict(issues),
    }
    summary_path = OUTPUT_DIR / 'check_dataset_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n===== Summary =====')
    for key in ('total', 'pass', 'warning', 'error'):
        print(f'{key}: {summary[key]}')
    print('\n===== Issue Counts =====')
    for issue, count in issues.most_common():
        print(f'{issue}: {count}')
    print('\n===== Problematic Songs (first 50) =====')
    problems = [r for r in results if r['status'] != 'PASS']
    for result in problems[:50]:
        print(result['song'], result['status'], result['issues'])
    if len(problems) > 50:
        print(f'... and {len(problems) - 50} more')
    print('\nCSV:', csv_path)
    print('Summary:', summary_path)


if __name__ == '__main__':
    main()

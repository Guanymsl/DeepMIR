import csv
import json
import pretty_midi
import numpy as np

from collections import Counter
from pathlib import Path
from miditok import REMI
from symusic import Track
from tqdm import tqdm

from tokenization_corrected import corrected_score, read_beats

DATA_DIR = Path("data/POP909/training")
TOKENIZER_PATH = Path("outputs/tokens/remi/tokenizer.json")
OUTPUT_DIR = Path("outputs")

MAX_TIME_ERROR_MS = 2.0
MAX_BAR_DIFFERENCE = 1
MAX_TEMPO_ERROR_MS = 2.0

FIELDS = [
    "song",
    "status",
    "notes_original",
    "notes_corrected",
    "pitch_velocity_match",
    "max_start_error_ms",
    "max_end_error_ms",
    "expected_bars",
    "token_bars",
    "bar_difference",
    "tokens",
    "unknown_tokens",
    "meters",
    "tempo_events",
    "max_beat_error_ms",
    "issues",
]

def get_original_notes(midi):
    notes = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            notes.append((
                note.pitch,
                note.velocity,
                note.start,
                note.end,
            ))

    return sorted(notes)

def get_corrected_notes(score):
    notes = []

    score_seconds = score.to(
        "second",
        min_dur=0.000001,
    )

    for track in score_seconds.tracks:
        if track.is_drum:
            continue

        for note in track.notes:
            notes.append((
                note.pitch,
                note.velocity,
                note.time,
                note.time + note.duration,
            ))

    return sorted(notes)

def compare_notes(original, corrected, result, issues):
    a = get_original_notes(original)
    b = get_corrected_notes(corrected)

    result["notes_original"] = len(a)
    result["notes_corrected"] = len(b)

    if len(a) != len(b):
        issues.append("NOTE_COUNT_MISMATCH")
        return

    a = sorted(a, key=lambda n: (n[0], n[1], n[2], n[3]))
    b = sorted(b, key=lambda n: (n[0], n[1], n[2], n[3]))

    match = all(
        x[:2] == y[:2]
        for x, y in zip(a, b)
    )

    result["pitch_velocity_match"] = match

    if not match:
        issues.append("PITCH_VELOCITY_MISMATCH")
        return

    if not a:
        return

    start_errors = np.array([
        abs(x[2] - y[2]) * 1000
        for x, y in zip(a, b)
    ])

    end_errors = np.array([
        abs(x[3] - y[3]) * 1000
        for x, y in zip(a, b)
    ])

    result["max_start_error_ms"] = float(start_errors.max())
    result["max_end_error_ms"] = float(end_errors.max())

    if start_errors.max() > MAX_TIME_ERROR_MS:
        issues.append("START_TIME_ERROR")

    if end_errors.max() > MAX_TIME_ERROR_MS:
        issues.append("END_TIME_ERROR")

def compare_beats(score, times, result, issues):

    score_seconds = score.to("second", min_dur=0.000001)

    result["tempo_events"] = len(score.tempos)

    if len(score.tempos) == 0:
        issues.append("NO_TEMPO")
        return

    # Build tick -> seconds mapping from tempo events
    tpq = score.ticks_per_quarter

    tempos = sorted(score.tempos, key=lambda t: t.time)

    tempo_ticks = [t.time for t in tempos]
    tempo_bpms = [t.qpm for t in tempos]

    def tick_to_seconds(tick):
        seconds = 0.0
        last_tick = 0
        bpm = 120.0

        for event_tick, event_bpm in zip(tempo_ticks, tempo_bpms):
            if event_tick > tick:
                break

            seconds += (
                (event_tick - last_tick)
                * 60.0
                / (tpq * bpm)
            )

            last_tick = event_tick
            bpm = event_bpm

        seconds += (
            (tick - last_tick)
            * 60.0
            / (tpq * bpm)
        )

        return seconds

    errors = [
        abs(tick_to_seconds(i * tpq) - expected) * 1000
        for i, expected in enumerate(times)
    ]

    result["max_beat_error_ms"] = float(max(errors))

    if max(errors) > MAX_TEMPO_ERROR_MS:
        issues.append("BEAT_TIME_ERROR")


def compare_meters(score, downbeats, result, issues):
    meters = [
        b - a
        for a, b in zip(downbeats, downbeats[1:])
    ]

    result["meters"] = ",".join(
        map(str, sorted(set(meters)))
    )

    if not meters:
        issues.append("NO_COMPLETE_BARS")
        return

    if any(m < 1 or m > 16 for m in meters):
        issues.append("INVALID_METER")
        return

    tpq = score.ticks_per_quarter

    actual = sorted(
        score.time_signatures,
        key=lambda ts: ts.time,
    )

    if not actual:
        issues.append("NO_TIME_SIGNATURE")
        return

    for beat_index, numerator in zip(downbeats, meters):
        tick = beat_index * tpq

        active = None

        for ts in actual:
            if ts.time <= tick:
                active = ts
            else:
                break

        if active is None:
            issues.append("MISSING_TIME_SIGNATURE")
            return

        if (
            active.numerator != numerator
            or active.denominator != 4
        ):
            issues.append("TIME_SIGNATURE_MISMATCH")
            return


def tokenize(score, tokenizer, result, issues):
    merged = Track(
        program=0,
        is_drum=False,
        name="PIANO_MERGED",
    )

    for track in score.tracks:
        if not track.is_drum:
            merged.notes.extend(track.notes)

    merged.notes.sort(
        key=lambda note: (
            note.time,
            note.pitch,
            note.duration,
        )
    )

    score.tracks = [merged]

    encoded = tokenizer.encode(score)
    sequences = (
        encoded
        if isinstance(encoded, list)
        else [encoded]
    )

    tokens = [
        token
        for sequence in sequences
        for token in sequence.tokens
    ]

    result["tokens"] = len(tokens)
    result["token_bars"] = tokens.count("Bar_None")

    result["unknown_tokens"] = sum(
        "UNK" in token.upper()
        for token in tokens
    )

    expected = result["expected_bars"]
    actual = result["token_bars"]

    result["bar_difference"] = actual - expected

    if abs(actual - expected) > MAX_BAR_DIFFERENCE:
        issues.append("BAR_COUNT_MISMATCH")

    if result["unknown_tokens"] > 0:
        issues.append("UNKNOWN_TOKENS")

    if len(tokens) == 0:
        issues.append("EMPTY_TOKENS")


def check_song(folder, tokenizer):
    name = folder.name

    midi_path = folder / f"{name}.mid"
    beat_path = folder / "beat_midi.txt"

    result = {field: None for field in FIELDS}
    result["song"] = name

    issues = []

    if not midi_path.is_file():
        raise FileNotFoundError(midi_path)

    if not beat_path.is_file():
        raise FileNotFoundError(beat_path)

    times, downbeats = read_beats(beat_path)

    if len(times) < 2:
        raise ValueError("Insufficient beats")

    if len(downbeats) < 2:
        raise ValueError("Insufficient downbeats")

    if not np.all(np.diff(times) > 0):
        raise ValueError("Beat times are not strictly increasing")

    if downbeats[0] != 0:
        issues.append("FIRST_DOWNBEAT_NOT_ZERO")

    if downbeats[-1] >= len(times):
        issues.append("INVALID_LAST_DOWNBEAT")

    result["expected_bars"] = len(downbeats) - 1

    original = pretty_midi.PrettyMIDI(str(midi_path))
    score, _ = corrected_score(midi_path, beat_path)

    compare_notes(original, score, result, issues)
    compare_beats(score, times, result, issues)
    compare_meters(score, downbeats, result, issues)
    tokenize(score, tokenizer, result, issues)

    result["status"] = "WARNING" if issues else "PASS"
    result["issues"] = ";".join(issues)

    return result

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(DATA_DIR)

    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(TOKENIZER_PATH)

    tokenizer = REMI(params=TOKENIZER_PATH)

    folders = sorted(
        folder
        for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    )

    results = []

    for folder in tqdm(folders, desc="Checking POP909"):
        try:
            result = check_song(folder, tokenizer)

        except Exception as error:
            result = {field: None for field in FIELDS}
            result["song"] = folder.name
            result["status"] = "ERROR"
            result["issues"] = (
                f"{type(error).__name__}: {error}"
            )

        results.append(result)

    csv_path = OUTPUT_DIR / "check_dataset.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(results)

    status_counts = Counter(
        result["status"]
        for result in results
    )

    issue_counts = Counter()

    for result in results:
        if result["status"] == "WARNING":
            issue_counts.update(
                result["issues"].split(";")
            )

    summary = {
        "total": len(results),
        "pass": status_counts["PASS"],
        "warning": status_counts["WARNING"],
        "error": status_counts["ERROR"],
        "issues": dict(issue_counts),
    }

    summary_path = OUTPUT_DIR / "check_dataset_summary.json"

    with summary_path.open("w") as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("\n===== Summary =====")
    print("Total:  ", summary["total"])
    print("Pass:   ", summary["pass"])
    print("Warning:", summary["warning"])
    print("Error:  ", summary["error"])

    print("\n===== Issue Counts =====")

    if issue_counts:
        for issue, count in issue_counts.most_common():
            print(f"{issue}: {count}")
    else:
        print("None")

    print("\n===== Problematic Songs =====")

    problematic = [
        r for r in results
        if r["status"] != "PASS"
    ]

    if not problematic:
        print("None")
    else:
        for result in problematic[:50]:
            print(
                result["song"],
                result["status"],
                result["issues"],
            )

        if len(problematic) > 50:
            print(
                f"... and {len(problematic) - 50} more"
            )

    print("\n===== Output =====")
    print("CSV:    ", csv_path)
    print("Summary:", summary_path)

if __name__ == "__main__":
    main()


from pathlib import Path
from collections import defaultdict
from statistics import mean, median

from symusic import Score

TASK_A_DIR = Path("outputs/generated/A")
TASK_B_DIR = Path("outputs/generated/B")


def analyze_midi(path):
    score = Score(str(path))
    events = []
    notes = []

    for track in score.tracks:
        for note in track.notes:
            start = note.time
            end = start + note.duration

            events.append((start, 1))
            events.append((end, -1))
            notes.append((start, end, note.pitch))

    events.sort(key=lambda x: (x[0], x[1]))

    active = 0
    max_polyphony = 0

    for _, change in events:
        active += change
        max_polyphony = max(max_polyphony, active)

    # Count overlapping notes of the same pitch
    by_pitch = defaultdict(list)

    for start, end, pitch in notes:
        by_pitch[pitch].append((start, end))

    same_pitch_overlaps = 0

    for intervals in by_pitch.values():
        intervals.sort()
        latest_end = -1

        for start, end in intervals:
            if start < latest_end:
                same_pitch_overlaps += 1
            latest_end = max(latest_end, end)

    return {
        "notes": len(notes),
        "max_polyphony": max_polyphony,
        "same_pitch_overlaps": same_pitch_overlaps,
    }


def analyze_folder(folder, name):
    files = sorted(folder.rglob("*.mid"))

    if not files:
        print(f"{name}: No MIDI files found")
        return

    results = []

    print(f"\n{name} ({len(files)} MIDI files)")
    print("-" * 75)

    for path in files:
        result = analyze_midi(path)
        results.append(result)

        print(
            f"{path.relative_to(folder)} | "
            f"Notes={result['notes']} | "
            f"MaxPoly={result['max_polyphony']} | "
            f"SamePitchOverlap={result['same_pitch_overlaps']}"
        )

    print("\nSummary:")
    print(f"Average Notes: {mean(r['notes'] for r in results):.1f}")
    print(f"Median Max Polyphony: {median(r['max_polyphony'] for r in results)}")
    print(f"Maximum Polyphony: {max(r['max_polyphony'] for r in results)}")
    print(f"Average Same-Pitch Overlaps: {mean(r['same_pitch_overlaps'] for r in results):.1f}")


def main():
    analyze_folder(TASK_A_DIR, "Task A")
    analyze_folder(TASK_B_DIR, "Task B")


if __name__ == "__main__":
    main()

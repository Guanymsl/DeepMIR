
from pathlib import Path
from collections import defaultdict
from statistics import mean, median
from symusic import Score
from tqdm import tqdm

TRAIN_DIR = Path("data/POP909/training")


def analyze_midi(path):
    score = Score(str(path))
    notes = []

    # 與 Training 相同：合併所有非 Drum Tracks
    for track in score.tracks:
        if not track.is_drum:
            for note in track.notes:
                notes.append((
                    note.time,
                    note.time + note.duration,
                    note.pitch,
                ))

    events = []
    by_pitch = defaultdict(list)

    for start, end, pitch in notes:
        events.append((start, 1))
        events.append((end, -1))
        by_pitch[pitch].append((start, end))

    events.sort(key=lambda x: (x[0], x[1]))

    active = 0
    max_polyphony = 0

    for _, change in events:
        active += change
        max_polyphony = max(max_polyphony, active)

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
        "overlap_ratio": same_pitch_overlaps / len(notes) if notes else 0,
    }


def main():
    files = sorted(TRAIN_DIR.rglob("*.mid"))
    results = []

    for path in tqdm(files, desc="Checking POP909"):
        result = analyze_midi(path)
        results.append(result)

    print(f"\nTotal MIDI: {len(results)}")

    for key in ("notes", "max_polyphony", "same_pitch_overlaps"):
        values = [r[key] for r in results]
        print(
            f"{key}: "
            f"mean={mean(values):.2f}, "
            f"median={median(values):.2f}, "
            f"max={max(values)}"
        )

    ratios = [r["overlap_ratio"] for r in results]
    print(f"Mean overlap ratio: {mean(ratios):.2%}")
    print(f"Median overlap ratio: {median(ratios):.2%}")

    print("\nTop 10 highest overlap ratios:")
    ranked = sorted(
        zip(files, results),
        key=lambda x: x[1]["overlap_ratio"],
        reverse=True,
    )

    for path, r in ranked[:10]:
        print(
            f"{path.parent.name}: "
            f"Notes={r['notes']} "
            f"MaxPoly={r['max_polyphony']} "
            f"Overlap={r['same_pitch_overlaps']} "
            f"Ratio={r['overlap_ratio']:.2%}"
        )


if __name__ == "__main__":
    main()

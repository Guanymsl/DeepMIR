
from pathlib import Path
from collections import defaultdict
import pretty_midi
from tokenization_corrected import corrected_score, read_beats

for name in ["005", "010", "013", "023"]:
    folder = Path("data/POP909/training") / name
    midi_path = folder / f"{name}.mid"
    beat_path = folder / "beat_midi.txt"

    original = pretty_midi.PrettyMIDI(str(midi_path))
    score, _ = corrected_score(midi_path, beat_path)
    converted = score.to("second", min_dur=0.000001)
    beats, _ = read_beats(beat_path)

    def collect(notes):
        groups = defaultdict(list)
        for pitch, velocity, start, end in notes:
            groups[(pitch, velocity)].append((start, end))
        for group in groups.values():
            group.sort()
        return groups

    a = collect(
        (n.pitch, n.velocity, n.start, n.end)
        for inst in original.instruments if not inst.is_drum
        for n in inst.notes
    )
    b = collect(
        (n.pitch, n.velocity, n.time, n.time + n.duration)
        for track in converted.tracks if not track.is_drum
        for n in track.notes
    )

    errors = []
    for key, originals in a.items():
        corrected = b.get(key, [])
        for old, new in zip(originals, corrected):
            errors.append((
                abs(old[0] - new[0]) * 1000,
                abs(old[1] - new[1]) * 1000,
                key,
                old,
                new,
            ))

    print(f"\n===== {name} =====")
    print("First beat:", beats[0])
    for start_err, end_err, key, old, new in sorted(
        errors, reverse=True
    )[:5]:
        print(
            f"pitch/velocity={key} "
            f"start_error={start_err:.3f}ms "
            f"end_error={end_err:.3f}ms"
        )
        print(f"  original: {old}")
        print(f"  corrected: {new}")

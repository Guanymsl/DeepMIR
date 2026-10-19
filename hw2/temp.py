from pathlib import Path
import pretty_midi
from tokenization_corrected import read_beats

for name in ["005", "010", "013", "023"]:
    folder = Path("data/POP909/training") / name
    midi = pretty_midi.PrettyMIDI(str(folder / f"{name}.mid"))
    beats, _ = read_beats(folder / "beat_midi.txt")

    starts = sorted(
        note.start
        for instrument in midi.instruments
        for note in instrument.notes
    )

    before = [t for t in starts if t < beats[0]]

    print(
        name,
        "first beat:", beats[0],
        "first note:", starts[0],
        "notes before first beat:", len(before),
    )
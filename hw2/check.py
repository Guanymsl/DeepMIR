
from pathlib import Path
from symusic import Score

GENERATED_DIR = Path("outputs/bonus/generated")

for path in sorted(GENERATED_DIR.glob("*/*.mid")):
    score = Score(str(path))
    tpq = score.ticks_per_quarter

    print(f"\nSong {path.parent.name}")

    for track in score.tracks:
        end_tick = max(
            (note.time + note.duration for note in track.notes),
            default=0,
        )

        bars = end_tick / (tpq * 4)

        print(f"  {track.name}: {bars:.2f} bars | {len(track.notes)} notes")

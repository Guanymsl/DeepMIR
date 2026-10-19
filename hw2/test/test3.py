from pathlib import Path
from miditok import REMI
from symusic import Score, Track

tokenizer = REMI(
    params=Path("outputs/tokens/remi/tokenizer.json")
)

score = Score("outputs/test_corrected/corrected.mid")

merged = Track(program=0, is_drum=False, name="PIANO_MERGED")

for track in score.tracks:
    if not track.is_drum:
        merged.notes.extend(track.notes)

merged.notes.sort(key=lambda n: (n.time, n.pitch, n.duration))
score.tracks = [merged]

encoded = tokenizer.encode(score)
sequences = encoded if isinstance(encoded, list) else [encoded]
tokens = [token for seq in sequences for token in seq.tokens]

print("Bar count:", tokens.count("Bar_None"))
print("Total tokens:", len(tokens))
print("Time signatures:", [
    token for token in tokens if token.startswith("TimeSig_")
])

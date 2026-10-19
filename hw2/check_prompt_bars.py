
from pathlib import Path

from miditok import REMI
from symusic import Track

from tokenization_corrected import corrected_score


TOKENIZER_PATH = Path("outputs/tokens/remi/tokenizer.json")
PROMPT_DIR = Path("data/POP909/prompts/basic-b")


def main():
    tokenizer = REMI(params=TOKENIZER_PATH)

    print(f"Tokenizer: {TOKENIZER_PATH}")

    for name in ["001", "002", "003"]:
        folder = PROMPT_DIR / name
        midi_path = folder / f"{name}.mid"
        beat_path = folder / "beat_midi.txt"

        score, changes = corrected_score(midi_path, beat_path)

        merged = Track(
            program=0,
            is_drum=False,
            name="PIANO_MERGED",
        )

        for track in score.tracks:
            if not track.is_drum:
                merged.notes.extend(track.notes)

        merged.notes.sort(
            key=lambda n: (n.time, n.pitch, n.duration)
        )

        score.tracks = [merged]

        encoded = tokenizer.encode(score)
        sequences = encoded if isinstance(encoded, list) else [encoded]

        tokens = [
            token
            for seq in sequences
            for token in seq.tokens
        ]

        bars = sum(token == "Bar_None" for token in tokens)
        time_signatures = [
            token for token in tokens
            if token.startswith("TimeSig_")
        ]

        print(
            f"{name}: "
            f"bars={bars}, "
            f"tokens={len(tokens)}, "
            f"time_signatures={time_signatures}, "
            f"changes={changes}, "
            f"expected=8, "
            f"match={bars == 8}"
        )


if __name__ == "__main__":
    main()

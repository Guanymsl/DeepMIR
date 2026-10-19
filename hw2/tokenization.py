import argparse
import json

from pathlib import Path
from tqdm import tqdm

from miditok import CPWord, REMI, TokenizerConfig
from symusic import Score, Track

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=["remi", "cpword"])
    return parser.parse_args()

def create(args):
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 16},
        num_velocities=32,
        use_tempos=True,
        use_time_signatures=False,
        use_programs=False,
        use_chords=False,
    )
    cls = REMI if args.type == "remi" else CPWord
    return cls(config)

def find(dataset_path):
    return [
        (folder.name, folder / f"{folder.name}.mid")
        for folder in sorted(dataset_path.iterdir())
        if folder.is_dir()
    ]

def normalize(encoded):
    return encoded if isinstance(encoded, list) else [encoded]

def main():
    args = parse_args()

    dataset_path = Path("data/POP909/training")
    output_dir = Path("outputs/tokens")

    tokenizer = create(args)
    dest = output_dir / args.type
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer.save(dest / "tokenizer.json")

    files = find(dataset_path)

    results = []
    for song_name, path in tqdm(files, desc=f"Tokenizing {args.type.upper()}", unit="song"):
        score = Score(str(path))
        merged = Track(program=0, is_drum=False, name="PIANO_MERGED")

        for track in score.tracks:
            if not track.is_drum:
                merged.notes.extend(track.notes)

        merged.notes.sort(key=lambda n: (n.time, n.pitch, n.duration))
        score.tracks = [merged]

        sequences = normalize(tokenizer.encode(score))
        results.append({
            "song_name": song_name,
            "source_midi": str(path),
            "sequences": [
                {"ids": seq.ids, "tokens": seq.tokens} for seq in sequences
            ],
        })

    with (dest / "tokens.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"Saved {len(results)} songs")

if __name__ == "__main__":
    main()
